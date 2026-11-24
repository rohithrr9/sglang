"""DFLASH mixed verify: row rules, the mixed batch, RNG order and the attention split."""

import dataclasses
import unittest
from types import SimpleNamespace
from typing import Optional

import torch

from sglang.srt.layers.attention import mixed_verify_attention as mva
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.dflash_mixed_verify import (
    DFlashMixedVerify,
    DraftBlock,
    MixedVerifyLayout,
    RngOrder,
    build_mixed_batch,
    host_prefix_bounds,
    measure_sampling_offset_step,
    pad_to_bucket,
    pass_refusal,
    verify_prune_lens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRowRules(CustomTestCase):
    def test_pad_to_bucket(self):
        self.assertEqual(pad_to_bucket(5, [4, 8, 16]), 8)
        self.assertEqual(pad_to_bucket(8, [4, 8, 16]), 8)
        self.assertEqual(pad_to_bucket(17, [4, 8, 16]), 17)
        self.assertEqual(pad_to_bucket(3, []), 3)

    def test_pass_refusal(self):
        buckets = [4096, 8192, 16384]
        ok = pass_refusal(
            num_extend_tokens=6000, num_verify_tokens=640, prefill_buckets=buckets
        )
        self.assertIsNone(ok)
        self.assertEqual(
            pass_refusal(
                num_extend_tokens=16000,
                num_verify_tokens=640,
                prefill_buckets=buckets,
            ),
            "above the largest prefill bucket",
        )
        # 1,840 rows pad to 4,096: more than 2x.
        self.assertEqual(
            pass_refusal(
                num_extend_tokens=1200, num_verify_tokens=640, prefill_buckets=buckets
            ),
            "pass off the prefill graph",
        )
        # 2,140 rows pad to 4,096: within 2x.
        self.assertIsNone(
            pass_refusal(
                num_extend_tokens=1500, num_verify_tokens=640, prefill_buckets=buckets
            )
        )
        self.assertIsNone(
            pass_refusal(
                num_extend_tokens=10**6, num_verify_tokens=8, prefill_buckets=[]
            )
        )

    def test_prune_lens_keep_every_verify_token(self):
        extend_lens, bs, block = [5, 3], 2, 4
        prune = verify_prune_lens(extend_lens, bs, block)
        self.assertEqual(prune, [5, 3] + [1] * 8)
        last_index = torch.cumsum(torch.tensor(prune), dim=0) - 1
        # The last token of each extend row, then every verify row.
        self.assertEqual(last_index.tolist(), [4, 7] + list(range(8, 16)))

    def test_host_prefix_bounds(self):
        bounds, why = host_prefix_bounds(
            reserved_kv_lens=torch.tensor([20, 9]), num_reqs=2, block_size=8
        )
        self.assertEqual((bounds, why), ([12, 1], None))
        self.assertEqual(
            host_prefix_bounds(reserved_kv_lens=None, num_reqs=2, block_size=8),
            (None, "no reserved lengths"),
        )
        self.assertEqual(
            host_prefix_bounds(
                reserved_kv_lens=torch.tensor([20]), num_reqs=2, block_size=8
            ),
            (None, "no reserved lengths"),
        )
        self.assertEqual(
            host_prefix_bounds(
                reserved_kv_lens=torch.tensor([20, 7]), num_reqs=2, block_size=8
            ),
            (None, "reserved length below the block"),
        )


class _FakeGenerator:
    def __init__(self):
        self.offset = 100

    def get_offset(self):
        return self.offset

    def set_offset(self, offset):
        self.offset = offset


class TestRngOrder(CustomTestCase):
    def test_serial_order(self):
        g = _FakeGenerator()
        rng = RngOrder(g, extend_step=4)
        rng.before_draft()
        self.assertEqual(g.offset, 104)  # the draft draws after the extend sampling
        g.offset += 12  # the draft's draws
        rng.before_extend_sampling()
        self.assertEqual(g.offset, 100)  # the extend sampling draws first
        g.offset += 4
        self.assertTrue(rng.after_extend_sampling())
        self.assertEqual(g.offset, 116)  # the verify draws after the draft's

    def test_mismatched_step_is_reported(self):
        g = _FakeGenerator()
        rng = RngOrder(g, extend_step=4)
        rng.before_draft()
        rng.before_extend_sampling()
        g.offset += 8
        self.assertFalse(rng.after_extend_sampling())

    def test_measured_once_and_restored(self):
        g = _FakeGenerator()
        calls = []

        def sampler(logits_output, sampling_info, *args):
            calls.append(logits_output.next_token_logits.shape)
            g.offset += 4

        info = SimpleNamespace(
            is_all_greedy=False,
            need_top_p_sampling=True,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            sampling_seed=None,
        )
        cache = {}
        for _ in range(2):
            step = measure_sampling_offset_step(
                sampler=sampler,
                sampling_info=info,
                num_reqs=3,
                vocab_size=11,
                device=torch.device("cpu"),
                generator=g,
                cache=cache,
            )
            self.assertEqual(step, 4)
        self.assertEqual(calls, [torch.Size([3, 11])])
        self.assertEqual(g.offset, 100)


def _draft_block(bs=2, block=4, prefix=(10, 20)) -> DraftBlock:
    prefix_lens = torch.tensor(prefix, dtype=torch.int64)
    positions_2d = prefix_lens[:, None] + torch.arange(block)
    slots_2d = 1000 + torch.arange(bs * block).view(bs, block)
    return DraftBlock(
        bs=bs,
        block_size=block,
        prefix_lens=prefix_lens,
        positions_2d=positions_2d,
        positions=positions_2d.reshape(-1),
        verify_out_cache_loc_2d=slots_2d,
        verify_out_cache_loc=slots_2d.reshape(-1),
        iteration_ids=None,
        draft_tokens=torch.arange(bs * block).view(bs, block) + 500,
        ring_block_locs=None,
    )


def _extend_batch():
    return SimpleNamespace(
        reqs=["e0", "e1"],
        input_ids=torch.arange(8),
        req_pool_indices=torch.tensor([3, 4]),
        req_pool_indices_cpu=torch.tensor([3, 4]),
        seq_lens=torch.tensor([5, 9]),
        seq_lens_cpu=torch.tensor([5, 9]),
        orig_seq_lens=None,
        seq_lens_sum=14,
        out_cache_loc=torch.arange(8) + 100,
        prefix_lens=[0, 6],
        extend_lens=[5, 3],
        extend_num_tokens=8,
        extend_logprob_start_lens=None,
        multimodal_inputs=None,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        return_logprob=False,
        is_prefill_only=False,
    )


class TestMixedBatch(CustomTestCase):
    def test_rows_appended(self):
        E = _extend_batch()
        R = SimpleNamespace(
            reqs=["r0", "r1"],
            req_pool_indices=torch.tensor([7, 8]),
            req_pool_indices_cpu=torch.tensor([7, 8]),
        )
        d = _draft_block()
        m = build_mixed_batch(
            extend_batch=E, running_batch=R, draft_block=d, verify_prefix_lens=[12, 21]
        )
        self.assertEqual(m.reqs, ["e0", "e1", "r0", "r1"])
        self.assertEqual(m.input_ids.tolist()[8:], list(range(500, 508)))
        self.assertEqual(m.req_pool_indices.tolist(), [3, 4, 7, 8])
        # Device lengths are exact; host lengths carry the given (bound) prefixes.
        self.assertEqual(m.seq_lens.tolist(), [5, 9, 14, 24])
        self.assertEqual(m.seq_lens_cpu.tolist(), [5, 9, 16, 25])
        self.assertEqual(m.seq_lens_sum, 55)
        self.assertEqual(m.out_cache_loc.tolist()[8:], list(range(1000, 1008)))
        self.assertEqual(m.prefix_lens, [0, 6, 12, 21])
        self.assertEqual(m.extend_lens, [5, 3, 4, 4])
        self.assertEqual(m.extend_num_tokens, 16)
        # The extend batch itself is untouched.
        self.assertEqual(E.reqs, ["e0", "e1"])
        self.assertEqual(E.extend_lens, [5, 3])


@dataclasses.dataclass
class _Metadata:
    cache_seqlens_int32: Optional[torch.Tensor] = None
    max_seq_len_q: int = 1
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    page_table: Optional[torch.Tensor] = None
    swa_page_table: Optional[torch.Tensor] = None
    zigzag_page_table: Optional[torch.Tensor] = None
    zigzag_swa_page_table: Optional[torch.Tensor] = None
    swa_out_cache_loc: Optional[torch.Tensor] = None
    is_ragged_verify: bool = False
    encoder_cache_seqlens: Optional[torch.Tensor] = None
    mixed_prefill_reqs: int = -1
    mixed_prefill_tokens: int = 0
    fmha_v2_copy_rows: Optional[torch.Tensor] = None
    fmha_v2_copy_cols: Optional[torch.Tensor] = None
    fmha_v2_block_table: Optional[torch.Tensor] = None


class _Backend:
    q_data_type = torch.float32

    def __init__(self, metadata):
        self.forward_metadata = metadata
        self.calls = []

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs):
        self.calls.append((q.shape[0], self.forward_metadata, forward_batch))
        return q * 2


def _pass(extend_lens=(5, 1), bs=2, block=4, bs_pad=3):
    n = len(extend_lens)
    P, T = sum(extend_lens), bs * block
    prefix = [0, 7]
    verify_prefix = [10, 20]
    seq_lens = [p + e for p, e in zip(prefix, extend_lens)] + [
        c + block for c in verify_prefix
    ]
    cache = torch.tensor(seq_lens, dtype=torch.int32)
    q_lens = torch.tensor(list(extend_lens) + [block] * bs, dtype=torch.int32)
    md = _Metadata(
        cache_seqlens_int32=cache,
        max_seq_len_q=block if max(extend_lens) < block else max(extend_lens),
        cu_seqlens_q=torch.nn.functional.pad(torch.cumsum(q_lens, 0), (1, 0)),
        cu_seqlens_k=torch.nn.functional.pad(torch.cumsum(cache, 0), (1, 0)),
        page_table=torch.arange((n + bs) * 3, dtype=torch.int32).view(n + bs, 3) + 1,
        swa_out_cache_loc=torch.arange(P + T) + 300,
    )
    fb = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=n + bs,
        out_cache_loc=torch.arange(P + T) + 100,
        seq_lens=torch.tensor(seq_lens),
        req_pool_indices=torch.tensor([1, 2, 7, 8]),
        extend_seq_lens=q_lens,
        extend_prefix_lens=torch.tensor(prefix + verify_prefix),
        extend_start_loc=torch.zeros(n + bs),
        seq_lens_cpu=torch.tensor(seq_lens),
        extend_seq_lens_cpu=list(extend_lens) + [block] * bs,
        extend_prefix_lens_cpu=prefix + verify_prefix,
        extend_logprob_start_lens_cpu=None,
        positions=torch.arange(P + T),
        spec_info=None,
        _attn_output=None,
    )
    layout = MixedVerifyLayout(
        num_extend_reqs=n,
        num_extend_tokens=P,
        extend_lens=list(extend_lens),
        num_verify_reqs=bs,
        num_padded_verify_reqs=bs_pad,
        block_size=block,
        prune_lens=torch.tensor(verify_prune_lens(extend_lens, bs, block)),
    )
    fb.mixed_verify = layout
    return md, fb, layout


class TestSplitViews(CustomTestCase):
    def test_views(self):
        md, fb, layout = _pass()
        views = mva.build_views(backend=_Backend(md), forward_batch=fb, layout=layout)
        mp, fp = views.extend_metadata, views.extend_batch
        self.assertEqual(mp.cache_seqlens_int32.tolist(), [5, 8])
        self.assertEqual(mp.cu_seqlens_q.tolist(), [0, 5, 6])
        self.assertEqual(mp.max_seq_len_q, 5)
        self.assertEqual(mp.page_table.shape[0], 2)
        self.assertEqual(mp.swa_out_cache_loc.tolist(), list(range(300, 306)))
        # A plain extend's one-token tail runs on the decode kernel, as stock splits it.
        self.assertEqual((mp.mixed_prefill_reqs, mp.mixed_prefill_tokens), (1, 5))
        self.assertEqual(fp.batch_size, 2)
        self.assertEqual(fp.out_cache_loc.tolist(), list(range(100, 106)))
        self.assertEqual(fp.positions.tolist(), list(range(6)))
        self.assertIsNone(fp.mixed_verify)
        self.assertIs(md.page_table.shape[0], 4)  # the pass's metadata is untouched

        mt, ft = views.verify_metadata, views.verify_batch
        self.assertIsInstance(mt, _Metadata)
        self.assertEqual(mt.cache_seqlens_int32.tolist(), [14, 24, 5])
        self.assertEqual(mt.max_seq_len_q, 4)
        self.assertEqual(mt.cu_seqlens_q.tolist(), [0, 4, 8, 12])
        self.assertEqual(mt.cu_seqlens_k.tolist(), [0, 14, 38, 43])
        self.assertEqual(mt.page_table[2].tolist(), [0, 0, 0])
        self.assertEqual(mt.swa_out_cache_loc.tolist()[:8], list(range(306, 314)))
        self.assertEqual(mt.swa_out_cache_loc.tolist()[8:], [0] * 4)
        self.assertEqual(mt.mixed_prefill_reqs, -1)
        self.assertEqual(ft.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertEqual(ft.batch_size, 3)
        self.assertEqual(ft.seq_lens.tolist(), [10, 20, 1])
        self.assertEqual(ft.req_pool_indices.tolist(), [7, 8, 0])
        self.assertEqual(ft.out_cache_loc.tolist(), list(range(106, 114)) + [0] * 4)
        self.assertEqual(ft.positions.tolist(), list(range(6, 14)) + [0] * 4)
        self.assertIsNone(ft.extend_seq_lens)
        self.assertIsNone(ft.spec_info)
        self.assertIsNone(ft.mixed_verify)

    def test_unsplittable_metadata(self):
        md, fb, layout = _pass()
        md.zigzag_page_table = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            mva.build_views(backend=_Backend(md), forward_batch=fb, layout=layout)

    def test_forward_split_and_assembly(self):
        md, fb, layout = _pass()
        backend = _Backend(md)
        rows = layout.num_extend_tokens + layout.num_verify_tokens
        q = torch.arange(rows * 2, dtype=torch.float32).view(rows, 2)
        out = mva.forward_mixed_verify(backend, q, q, q, None, fb, True)
        self.assertTrue(torch.equal(out, q * 2))
        (n_e, md_e, fb_e), (n_v, md_v, fb_v) = backend.calls
        self.assertEqual((n_e, n_v), (6, 12))  # verify rows padded to bs_pad blocks
        self.assertEqual(fb_e.forward_mode, ForwardMode.EXTEND)
        self.assertEqual(fb_v.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertIs(md_v, layout.attn_views.verify_metadata)
        self.assertIs(backend.forward_metadata, md)  # restored
        # One view build per pass.
        views = layout.attn_views
        mva.forward_mixed_verify(backend, q, q, q, None, fb, True)
        self.assertIs(layout.attn_views, views)

    def test_decode_tail_split(self):
        self.assertEqual(mva.decode_tail_split([5, 1, 1]), (1, 5))
        self.assertEqual(mva.decode_tail_split([1, 1]), (0, 0))
        self.assertEqual(mva.decode_tail_split([5, 3]), (-1, 0))


def _running(n):
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

    spec = DFlashDraftInputV2.__new__(DFlashDraftInputV2)
    return SimpleNamespace(
        reqs=[
            SimpleNamespace(return_routed_experts=False, beam_group=None)
            for _ in range(n)
        ],
        is_prefill_only=False,
        has_grammar=False,
        sampling_info=None,
        spec_info=spec,
        is_empty=lambda: n == 0,
    )


def _extend(extend_lens):
    return SimpleNamespace(
        reqs=[
            SimpleNamespace(return_routed_experts=False, beam_group=None)
            for _ in extend_lens
        ],
        forward_mode=ForwardMode.EXTEND,
        is_prefill_only=False,
        has_grammar=False,
        return_logprob=False,
        sampling_info=None,
        input_embeds=None,
        multimodal_inputs=None,
        extend_lens=list(extend_lens),
        prefix_lens=[0] * len(extend_lens),
    )


class TestCollaborator(CustomTestCase):
    def _make(self):
        return DFlashMixedVerify(
            block_size=8,
            prefill_buckets=[16384, 4096, 8192],
            decode_buckets=[1, 2, 4, 8],
            max_verify_reqs=9,
        )

    def test_reservation(self):
        mv = self._make()
        self.assertEqual(mv.reserved_tokens(_running(3)), 24)
        self.assertEqual(mv.reserved_tokens(_running(0)), 0)
        grammar = _running(3)
        grammar.has_grammar = True
        self.assertEqual(mv.reserved_tokens(grammar), 0)

    def test_fold_once_for_its_batch(self):
        mv = self._make()
        E, R = _extend([3000, 1000]), _running(4)
        prepared = []
        mv.maybe_fold(
            extend_batch=E,
            running_batch=R,
            prepare_decode=lambda b: prepared.append(b) or b,
        )
        self.assertEqual(prepared, [R])
        self.assertIsNone(mv.take_fold(_extend([1])))
        mv.maybe_fold(extend_batch=E, running_batch=R, prepare_decode=lambda b: b)
        self.assertIs(mv.take_fold(E), R)
        self.assertIsNone(mv.take_fold(E))

    def test_refusals_do_not_prepare_decode(self):
        mv = self._make()
        prepared = []
        mv.maybe_fold(
            extend_batch=_extend([1000]),  # 1,032 rows pad to 4,096: off the graph
            running_batch=_running(4),
            prepare_decode=prepared.append,
        )
        mv.maybe_fold(
            extend_batch=_extend([4000]),
            running_batch=_running(16),  # pads past the verify request limit
            prepare_decode=prepared.append,
        )
        self.assertEqual(prepared, [])
        self.assertEqual(
            dict(mv.refusals),
            {
                "pass off the prefill graph": 1,
                "running batch above the verify request limit": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
