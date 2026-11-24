"""Kernels that fold launches of the DFLASH decode step: each must write exactly the
bytes of the torch ops it replaces, and nothing outside them."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.dflash import stage_sampling_params
from sglang.kernels.ops.speculative.dflash_accept_tail import dflash_accept_tail
from sglang.srt.managers.overlap_utils import (
    FutureMap,
    FutureRelayBuffers,
    RelayPayload,
    offer_future_relay_buffers,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative import dflash_worker_v2 as worker_mod
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestAcceptTail(CustomTestCase):
    # The same non-zero pattern in both relays: a row nobody writes must match too.
    POISON = 0x5A5A5A5A5A5A5A5A

    def setUp(self):
        # stash reads the published speculative algorithm.
        override = get_context().override_server_args(speculative_algorithm="DFLASH")
        override.install()
        self.addCleanup(override.restore)
        self.addCleanup(offer_future_relay_buffers, None)

    def _future_map(self, pool, relay):
        offer_future_relay_buffers(relay)
        future_map = FutureMap(
            device=torch.device("cuda"),
            spec_algo=SpeculativeAlgorithm.DFLASH,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((pool, 1), device="cuda")
            ),
            needs_cpu_seq_lens=False,
        )
        future_map.new_seq_lens_buf.fill_(self.POISON)
        future_map.output_tokens_buf.fill_(self.POISON)
        return future_map

    def test_one_launch_leaves_the_relay_publish_and_stash_leave(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        pool, block = 4096, 8
        for bs in (1, 2, 3, 5, 17, 64, 128, 129, 300):
            prefix = torch.randint(
                100_000, 250_000, (bs,), generator=gen, device="cuda"
            )
            commit = torch.randint(
                1, block + 1, (bs,), generator=gen, device="cuda", dtype=torch.int32
            )
            bonus = torch.randint(0, 202048, (bs,), generator=gen, device="cuda")
            rows = torch.randperm(pool - 1, generator=gen, device="cuda")[:bs] + 1

            stock = self._future_map(pool, None)
            want = prefix + commit.to(prefix.dtype)
            stock.publish(rows, want)
            stock.stash(rows, RelayPayload(bonus_tokens=bonus))

            relay = FutureRelayBuffers(req_pool_size=pool, device="cuda")
            fused = self._future_map(pool, relay)
            got = dflash_accept_tail(
                prefix_lens=prefix,
                commit_lens=commit,
                bonus=bonus,
                indices=rows,
                new_seq_lens_buf=relay.new_seq_lens,
                output_tokens_buf=relay.output_tokens,
            )
            relay.published_seq_lens, relay.published_bonus = got, bonus
            fused.publish(rows, got)
            fused.stash(rows, RelayPayload(bonus_tokens=bonus))
            with self.subTest(bs=bs):
                self.assertTrue(relay.adopted)
                self.assertTrue(torch.equal(got, want))
                self.assertTrue(
                    torch.equal(fused.new_seq_lens_buf, stock.new_seq_lens_buf)
                )
                self.assertTrue(
                    torch.equal(fused.output_tokens_buf, stock.output_tokens_buf)
                )
                # Both scatters were skipped, so the rows above are the kernel's.
                self.assertIsNone(relay.published_seq_lens)
                self.assertIsNone(relay.published_bonus)


if __name__ == "__main__":
    unittest.main()
