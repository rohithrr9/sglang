"""Kernels that fold launches of the DFLASH decode step: each must write exactly the
bytes of the torch ops it replaces, and nothing outside them."""

import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.dflash import (
    _prepare_dflash_draft_block_unchecked,
    stage_sampling_params,
)
from sglang.kernels.ops.speculative.dflash_accept_tail import dflash_accept_tail
from sglang.srt.environ import envs
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers import overlap_utils
from sglang.srt.managers.overlap_utils import (
    FutureMap,
    FutureRelayBuffers,
    RelayPayload,
    offer_future_relay_buffers,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.muse_spark_omni import _NormalizedEmbedding
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative import dflash_worker_v2 as worker_mod
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype == torch.float32:
        a, b = a.view(torch.int32), b.view(torch.int32)
    return a.dtype == b.dtype and torch.equal(a, b)


def _bind(worker, *names):
    for name in names:
        setattr(
            worker, name, MethodType(getattr(worker_mod.DFlashWorkerV2, name), worker)
        )
    return worker


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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestForkPrep(CustomTestCase):
    # Hidden spans two column blocks of the row gather, the second one partial.
    POOL, WIDTH, BLOCK, MASK, VOCAB, HIDDEN = 512, 1024, 8, 5, 1000, 1088

    def setUp(self):
        enter_scope(self, published_topology("test", tp_size=1, pp_size=1))
        enter_scope(self, get_parallel().override(tp_group=None))
        # CI runs with the relay's debug asserts, which keep the resolve's gather.
        enter_scope(self, mock.patch.object(overlap_utils, "_DEBUG_ASSERT", False))

    def _table(self, dtype, **kwargs):
        table = VocabParallelEmbedding(
            self.VOCAB, self.HIDDEN, params_dtype=dtype, **kwargs
        ).cuda()
        with torch.no_grad():
            table.weight.normal_(generator=torch.Generator("cuda").manual_seed(0))
        return table

    def _future_map(self, *, fused, relay_tokens):
        with envs.SGLANG_OPT_DFLASH_FUSED_FORK_PREP.override(fused):
            future_map = FutureMap(
                device=torch.device("cuda"),
                spec_algo=SpeculativeAlgorithm.DFLASH,
                req_to_token_pool=SimpleNamespace(
                    req_to_token=torch.zeros((self.POOL, 1), device="cuda")
                ),
                needs_cpu_seq_lens=False,
            )
        future_map.output_tokens_buf.copy_(relay_tokens)
        return future_map

    def _resolve(self, future_map, future_indices):
        draft_input = DFlashDraftInputV2.create_idle_input(device="cuda")
        draft_input.future_indices = future_indices
        future_map._resolve_spec_extras(
            SimpleNamespace(
                spec_info=draft_input,
                forward_mode=ForwardMode.DECODE,
                is_extend_in_batch=False,
            )
        )
        return draft_input

    def _block_buffers(self, bs):
        # Two spare rows past the batch: the launch must not write them.
        return [
            torch.full((bs + 2, self.BLOCK), -9, dtype=torch.int64, device="cuda")
            for _ in range(3)
        ]

    def test_resolve_and_one_launch_write_the_stock_gather_block_and_rows(self):
        gen = torch.Generator("cuda").manual_seed(0)
        tables = {
            dtype: self._table(dtype) for dtype in (torch.bfloat16, torch.float32)
        }
        for bs in (1, 2, 3, 5, 17, 64, 129):
            req_to_token = torch.randint(
                0, 1 << 30, (self.POOL, self.WIDTH), generator=gen, device="cuda"
            ).to(torch.int32)
            relay_tokens = torch.randint(
                0, self.VOCAB, (self.POOL,), generator=gen, device="cuda"
            )
            req_pool_indices = torch.randperm(self.POOL, generator=gen, device="cuda")[
                :bs
            ]
            future_indices = torch.randperm(self.POOL, generator=gen, device="cuda")[
                :bs
            ]
            prefix_lens = torch.randint(
                0, self.WIDTH, (bs,), generator=gen, device="cuda"
            )
            # A block running past the req_to_token row reads slot 0 there.
            prefix_lens[0] = self.WIDTH - 3
            stock_input = self._resolve(
                self._future_map(fused=False, relay_tokens=relay_tokens),
                future_indices,
            )
            want = self._block_buffers(bs)
            _prepare_dflash_draft_block_unchecked(
                bonus_tokens=stock_input.bonus_tokens,
                prefix_lens=prefix_lens,
                req_pool_indices=req_pool_indices,
                req_to_token=req_to_token,
                block_ids_out=want[0][:bs],
                positions_out=want[1][:bs],
                cache_loc_out=want[2][:bs],
                mask_token_id=self.MASK,
            )
            worker = _bind(
                SimpleNamespace(
                    model_runner=SimpleNamespace(
                        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token)
                    ),
                    _mask_token_id=self.MASK,
                    _runtime=None,
                    _use_triton_prepare_block=True,
                ),
                "_take_bonus_tokens_source",
                "_prepare_block_from_relay",
            )
            cases = [("no rows", None, None, torch.int64)]
            for dtype, table in tables.items():
                wrapper = _NormalizedEmbedding(table, torch.nn.Identity())
                cases += [(f"{dtype} table", table, table, torch.int32)]
                cases += [(f"{dtype} wrapper", wrapper, table, torch.int64)]
            for name, embedding, table, index_dtype in cases:
                fused_map = self._future_map(fused=True, relay_tokens=relay_tokens)
                fused_input = self._resolve(fused_map, future_indices.to(index_dtype))
                with self.subTest(bs=bs, case=name):
                    self.assertIs(
                        fused_input.bonus_tokens_source, fused_map.output_tokens_buf
                    )
                    source = worker._take_bonus_tokens_source(fused_input)
                    self.assertIs(source, fused_map.output_tokens_buf)
                    got = self._block_buffers(bs)
                    rows = worker._prepare_block_from_relay(
                        bonus_tokens_source=source,
                        draft_input=fused_input,
                        embedding=embedding,
                        prefix_lens=prefix_lens,
                        req_pool_indices=req_pool_indices,
                        block_ids=got[0][:bs],
                        positions_2d=got[1][:bs],
                        verify_out_cache_loc_2d=got[2][:bs],
                    )
                    self.assertTrue(
                        torch.equal(fused_input.bonus_tokens, stock_input.bonus_tokens)
                    )
                    for got_buf, want_buf in zip(got, want):
                        self.assertTrue(torch.equal(got_buf, want_buf))
                    if table is None:
                        self.assertIsNone(rows)
                    else:
                        self.assertTrue(torch.equal(rows, table(want[0][:bs])))

    def test_fold_takes_only_one_whole_unquantized_table(self):
        def resolve(module, **fields):
            model = SimpleNamespace(get_input_embeddings=lambda: module)
            worker = SimpleNamespace(
                _full_embed_gpu=None,
                draft_model=SimpleNamespace(),
                _target_worker=SimpleNamespace(
                    model_runner=SimpleNamespace(model=model)
                ),
            )
            for name, value in fields.items():
                setattr(worker, name, value)
            return _bind(
                worker, "_resolve_fork_prep_embedding"
            )._resolve_fork_prep_embedding()

        table = self._table(torch.bfloat16)
        wrapper = _NormalizedEmbedding(table, torch.nn.Identity())
        self.assertIs(resolve(table), table)
        self.assertIs(resolve(wrapper), wrapper)
        half = self._table(torch.float16)
        self.assertIs(resolve(half), half)
        plain = torch.nn.Embedding(8, 4).cuda()
        misses = {
            "replicated table": (table, dict(_full_embed_gpu=table.weight)),
            "plain embedding": (plain, {}),
            "wrapper of a plain embedding": (
                _NormalizedEmbedding(plain, torch.nn.Identity()),
                {},
            ),
            "lm head": (ParallelLMHead(self.VOCAB, self.HIDDEN).cuda(), {}),
            "table on the host": (VocabParallelEmbedding(self.VOCAB, self.HIDDEN), {}),
            "output cast": (
                self._table(torch.bfloat16, output_dtype=torch.float32),
                {},
            ),
        }
        rows = table.weight.shape[0]
        for name, attr, value in (
            ("sharded", "tp_size", 2),
            ("added vocabulary", "num_added_embeddings", 1),
            ("quantized", "quant_method", SimpleNamespace()),
            (
                "integer table",
                "weight",
                torch.nn.Parameter(
                    torch.zeros((rows, 4), dtype=torch.int8, device="cuda"),
                    requires_grad=False,
                ),
            ),
            (
                "strided table",
                "weight",
                torch.nn.Parameter(
                    torch.zeros((4, rows), device="cuda").t(), requires_grad=False
                ),
            ),
        ):
            changed = self._table(torch.bfloat16)
            setattr(changed, attr, value)
            misses[name] = (changed, {})
        for name, (module, fields) in misses.items():
            with self.subTest(miss=name):
                self.assertIsNone(resolve(module, **fields))


if __name__ == "__main__":
    unittest.main()
