"""DFLASH decode-step launch folds, against stub workers and CPU modules: a fold may
drop launches, never change a value the step reads."""

import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.muse_spark.muse_spark_layers import MuseSpark1xRMSNorm
from sglang.srt.layers.vocab_parallel_embedding import (
    RowGatherEmbedding,
    VocabParallelEmbedding,
)
from sglang.srt.managers import overlap_utils
from sglang.srt.managers.overlap_utils import FutureMap, FutureRelayBuffers
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.muse_spark_omni import _NormalizedEmbedding
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _bind(worker, *names, cls=DFlashWorkerV2):
    for name in names:
        setattr(worker, name, MethodType(getattr(cls, name), worker))
    return worker


class TestDraftTokensAlias(CustomTestCase):
    def test_aliased_block_matches_the_copied_block_across_growth(self):
        """With the tokens buffer aliased, the verify block keeps the bonus column the
        block prep wrote, also after the buffers grow."""
        block_size, mask_id = 4, 7
        gen = torch.Generator().manual_seed(0)
        workers = {
            alias: _bind(
                SimpleNamespace(
                    block_size=block_size,
                    device="cpu",
                    _alias_draft_tokens=alias,
                    _batched_small_copy=False,
                    _draft_block_ids_buf=None,
                ),
                "_ensure_draft_block_buffers",
                "_assemble_draft_tokens",
            )
            for alias in (False, True)
        }
        for bs in (1, 2, 3, 4, 5, 8, 16, 3, 64, 2):
            bonus = torch.randint(0, 1000, (bs,), generator=gen)
            draft_next = torch.randint(0, 1000, (bs, block_size - 1), generator=gen)
            blocks = {}
            for alias, w in workers.items():
                w._ensure_draft_block_buffers(bs)
                block_ids = w._draft_block_ids_buf[:bs]
                block_ids.fill_(mask_id)
                block_ids[:, 0].copy_(bonus)
                blocks[alias] = w._assemble_draft_tokens(
                    bs=bs, block_ids=block_ids, draft_next=draft_next
                ).clone()
            with self.subTest(bs=bs):
                self.assertTrue(torch.equal(blocks[True], blocks[False]))
                self.assertTrue(torch.equal(blocks[True][:, 0], bonus))
                self.assertTrue(torch.equal(blocks[True][:, 1:], draft_next))


class TestFusedAcceptTailGate(CustomTestCase):
    def test_engages_only_where_the_kernel_writes_each_row_once(self):
        """The relay's publish and stash skip what the kernel wrote, so a step it
        cannot write exactly (duplicate or foreign rows, other dtypes, lengths
        already built, no overlap publish) must keep the stock tail."""
        relay = FutureRelayBuffers(req_pool_size=8, device="cpu")
        relay.adopted = True
        worker = _bind(
            SimpleNamespace(_accept_tail_relay=relay, _need_mamba_verify_commit=False),
            "_fused_accept_tail_relay",
        )

        def call(*, host=(3, 5, 1), beam_tail=None, **overrides):
            rows = torch.tensor([3, 5, 1])
            kwargs = dict(
                batch=SimpleNamespace(
                    req_pool_indices=rows,
                    req_pool_indices_cpu=torch.tensor(host),
                    beam_tail=beam_tail,
                ),
                on_publish=print,
                new_seq_lens=None,
                prefix_lens=torch.tensor([10, 20, 30]),
                commit_lens=torch.tensor([1, 2, 3], dtype=torch.int32),
                bonus=torch.tensor([7, 8, 9]),
            )
            kwargs.update(overrides)
            return worker._fused_accept_tail_relay(**kwargs)

        self.assertIs(call(), relay)
        misses = {
            "no overlap publish": dict(on_publish=None),
            "lengths already built": dict(new_seq_lens=torch.tensor([11, 22, 33])),
            "int64 commit lengths": dict(commit_lens=torch.tensor([1, 2, 3])),
            "int32 bonus": dict(bonus=torch.tensor([7, 8, 9], dtype=torch.int32)),
            "duplicate rows": dict(host=(3, 3, 1)),
            "row past the relay": dict(host=(3, 8, 1)),
            "beam tail": dict(beam_tail=object()),
        }
        for name, overrides in misses.items():
            with self.subTest(miss=name):
                self.assertIsNone(call(**overrides))
        for name, attr, value in (
            ("mamba commit", "_need_mamba_verify_commit", True),
            ("relay not adopted", "adopted", False),
        ):
            target = relay if attr == "adopted" else worker
            setattr(target, attr, value)
            with self.subTest(miss=name):
                self.assertIsNone(call())
            setattr(target, attr, not value)


class TestFusedForkPrepDeferral(CustomTestCase):
    def test_knob_arms_only_the_dflash_relay(self):
        for algo in ("DFLASH", "DSPARK", "EAGLE"):
            override = get_context().override_server_args(speculative_algorithm=algo)
            override.install()
            try:
                for knob in (False, True):
                    with envs.SGLANG_OPT_DFLASH_FUSED_FORK_PREP.override(knob):
                        future_map = FutureMap(
                            device=torch.device("cpu"),
                            spec_algo=SpeculativeAlgorithm.from_string(algo),
                            req_to_token_pool=SimpleNamespace(
                                req_to_token=torch.zeros((8, 1))
                            ),
                            needs_cpu_seq_lens=False,
                        )
                    with self.subTest(algo=algo, knob=knob):
                        self.assertEqual(
                            future_map._fused_fork_prep, knob and algo == "DFLASH"
                        )
            finally:
                override.restore()

    def test_defers_only_a_decode_batch_relaying_bonus_tokens(self):
        """The worker's fork prep reads the relay rows itself; any other batch, or a
        relay carrying more than bonus tokens, keeps the resolve's gather."""
        cuda = torch.device("cuda", 0)
        future_map = _bind(
            SimpleNamespace(
                _fused_fork_prep=True,
                need_hidden_states=False,
                output_tokens_buf=SimpleNamespace(device=cuda),
            ),
            "_defers_bonus_gather",
            cls=FutureMap,
        )

        def defers(*, mode=ForwardMode.DECODE, extend_in_batch=False, dsa=False, **ix):
            batch = SimpleNamespace(
                forward_mode=mode,
                is_extend_in_batch=extend_in_batch,
                spec_info=SimpleNamespace(future_dsa_topk_indices_available=dsa),
            )
            indices = SimpleNamespace(
                **{**dict(is_cuda=True, dtype=torch.int64, device=cuda), **ix}
            )
            return future_map._defers_bonus_gather(batch, indices)

        with mock.patch.object(overlap_utils, "_DEBUG_ASSERT", False):
            self.assertTrue(defers())
            self.assertTrue(defers(dtype=torch.int32))
            misses = {
                "extend": dict(mode=ForwardMode.EXTEND),
                "idle": dict(mode=ForwardMode.IDLE),
                "extend in batch": dict(extend_in_batch=True),
                "dsa top-k relayed": dict(dsa=True),
                "cpu indices": dict(is_cuda=False, device=torch.device("cpu")),
                "int16 indices": dict(dtype=torch.int16),
                "indices on another device": dict(device=torch.device("cuda", 1)),
            }
            for name, overrides in misses.items():
                with self.subTest(miss=name):
                    self.assertFalse(defers(**overrides))
            for attr, value in (
                ("_fused_fork_prep", False),
                ("need_hidden_states", True),
            ):
                setattr(future_map, attr, value)
                with self.subTest(miss=attr):
                    self.assertFalse(defers())
                setattr(future_map, attr, not value)
        with mock.patch.object(overlap_utils, "_DEBUG_ASSERT", True):
            self.assertFalse(defers())

    def test_worker_gathers_unless_the_triton_prep_reads_the_relay(self):
        relay = torch.arange(100, 132)
        for runtime, triton_prep, index_dtype in (
            (None, True, torch.int64),
            (None, True, torch.int32),
            (object(), True, torch.int64),
            (None, False, torch.int32),
        ):
            worker = _bind(
                SimpleNamespace(
                    _runtime=runtime, _use_triton_prepare_block=triton_prep
                ),
                "_take_bonus_tokens_source",
            )
            indices = torch.tensor([7, 0, 31, 7], dtype=index_dtype)
            draft_input = SimpleNamespace(
                bonus_tokens_source=relay,
                future_indices=indices,
                bonus_tokens=torch.full((4,), -1),
            )
            taken = worker._take_bonus_tokens_source(draft_input)
            fused = runtime is None and triton_prep
            with self.subTest(
                runtime=runtime, triton_prep=triton_prep, dtype=index_dtype
            ):
                self.assertIsNone(draft_input.bonus_tokens_source)
                if fused:
                    self.assertIs(taken, relay)
                    self.assertTrue(torch.all(draft_input.bonus_tokens == -1))
                else:
                    self.assertIsNone(taken)
                    self.assertTrue(
                        torch.equal(draft_input.bonus_tokens, relay[indices.long()])
                    )
        draft_input = SimpleNamespace(bonus_tokens_source=None)
        self.assertIsNone(worker._take_bonus_tokens_source(draft_input))


class TestRowGatherEmbedding(CustomTestCase):
    VOCAB, HIDDEN = 40, 16

    def setUp(self):
        enter_scope(self, published_topology("test", tp_size=1, pp_size=1))
        enter_scope(self, get_parallel().override(tp_group=None))

    def _table(self, **kwargs):
        table = VocabParallelEmbedding(self.VOCAB, self.HIDDEN, **kwargs)
        with torch.no_grad():
            table.weight.normal_(generator=torch.Generator().manual_seed(0))
        return table

    def test_rows_give_the_forward_of_the_table_and_the_muse_wrapper(self):
        ids = torch.tensor([[3, 39, 0, 0], [17, 17, 5, 39]])
        norm = MuseSpark1xRMSNorm(self.HIDDEN)
        for kwargs in (
            {},
            dict(params_dtype=torch.bfloat16),
            dict(output_dtype=torch.float16),
        ):
            table = self._table(**kwargs)
            rows = torch.nn.functional.embedding(ids, table.weight)
            wrapper = _NormalizedEmbedding(table, norm)
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                self.assertIsInstance(table, RowGatherEmbedding)
                self.assertIs(table.row_table, table)
                self.assertTrue(
                    torch.equal(table.forward_from_rows(ids, rows), table(ids))
                )
                self.assertIsInstance(wrapper, RowGatherEmbedding)
                self.assertIs(wrapper.row_table, table)
                self.assertTrue(
                    torch.equal(wrapper.forward_from_rows(ids, rows), wrapper(ids))
                )

    def test_no_row_table_without_one_whole_table(self):
        wrapper = _NormalizedEmbedding(
            torch.nn.Embedding(self.VOCAB, self.HIDDEN), MuseSpark1xRMSNorm(self.HIDDEN)
        )
        self.assertIsNone(wrapper.row_table)
        self.assertNotIsInstance(
            torch.nn.Embedding(self.VOCAB, self.HIDDEN), RowGatherEmbedding
        )
        table = self._table()
        table.tp_size = 2
        with self.assertRaises(ValueError):
            table.forward_from_rows(torch.zeros(1, dtype=torch.long), table.weight[:1])


if __name__ == "__main__":
    unittest.main()
