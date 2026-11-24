"""Kernels that fold launches of the DFLASH decode step: each must write exactly the
bytes of the torch ops it replaces, and nothing outside them."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.dflash import stage_sampling_params
from sglang.srt.speculative import dflash_worker_v2 as worker_mod
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype == torch.float32:
        a, b = a.view(torch.int32), b.view(torch.int32)
    return a.dtype == b.dtype and torch.equal(a, b)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestSamplerStage(CustomTestCase):
    # Below, at and above the bound, NaN, infinities and signed zeros.
    SPECIAL = (float("nan"), 0.0, -0.0, 1e-6, 1e-5, 2e-5, -1.0, 0.7, float("inf"))

    def _params(self, bs, gen):
        temps = torch.rand((bs, 1), generator=gen, device="cuda") * 2
        flat = temps.view(-1)
        for i, v in enumerate(self.SPECIAL[:bs]):
            flat[i] = v
        top_ks = torch.randint(-1, 4, (bs,), generator=gen, device="cuda")
        top_ks[: min(bs, 2)] = 1 << 30
        return temps, top_ks.to(torch.int32)

    def test_kernel_matches_clamp_and_mask(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        for bs in (1, 3, 9, 16, 17, 64, 300):
            temps, top_ks = self._params(bs, gen)
            t_out = torch.full((bs + 5,), -7.0, device="cuda")
            g_out = torch.zeros((bs + 5,), dtype=torch.bool, device="cuda")
            stage_sampling_params(
                temps.view(-1)[:bs], top_ks, t_out[:bs], g_out[:bs], 1e-5
            )
            want_t = torch.clamp(temps.view(-1)[:bs], min=1e-5)
            with self.subTest(bs=bs):
                self.assertTrue(_same_bits(t_out[:bs], want_t))
                self.assertTrue(torch.equal(g_out[:bs], top_ks <= 1))
                self.assertTrue(torch.all(t_out[bs:] == -7.0))
                self.assertFalse(g_out[bs:].any())

    def _sampler(self, one_launch):
        sampler = worker_mod._SelectorDraftSampler.__new__(
            worker_mod._SelectorDraftSampler
        )
        sampler.temperatures = torch.full((8,), 3.0, device="cuda")
        sampler.greedy_mask = torch.zeros(8, dtype=torch.bool, device="cuda")
        sampler.sampling_enabled = True
        sampler._stage_in_one_launch = one_launch
        return sampler

    def test_sampler_stages_the_same_buffers_or_declines(self):
        """A layout the kernel does not read (int64 top_ks, a float64 temperature
        column) must take the torch path, not a misread one."""
        gen = torch.Generator(device="cuda").manual_seed(1)
        bs = 6
        temps, top_ks = self._params(bs, gen)
        cases = {
            "fused": (temps, top_ks),
            "int64 top_ks": (temps, top_ks.to(torch.int64)),
            "float64 temperatures": (temps.double(), top_ks),
        }
        for name, (t, k) in cases.items():
            info = SimpleNamespace(temperatures=t, top_ks=k)
            got, want = self._sampler(True), self._sampler(False)
            got.stage_sampling_params(bs=bs, sampling_info=info)
            want.stage_sampling_params(bs=bs, sampling_info=info)
            with self.subTest(case=name):
                self.assertTrue(_same_bits(got.temperatures, want.temperatures))
                self.assertTrue(torch.equal(got.greedy_mask, want.greedy_mask))


if __name__ == "__main__":
    unittest.main()
