"""Decode page addressing and row-tail policy are independent contracts."""

import itertools
import unittest

import torch

from sglang.kernels.ops.attention.metadata import (
    fill_slots_then_decode_metadata,
    normal_decode_set_metadata,
)
from sglang.kernels.ops.memory.slot_fill import fill_slots
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-small")

POISON = -0x3C3C3C3C


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDecodeMetadataMapping(CustomTestCase):
    def test_mapping_and_tail_policy(self):
        # Cross a whole column block of dead pages, so zero_tail must also
        # control the early exit rather than just the last block's store mask.
        for ps, mapped_full, swa_kind, zero_tail in itertools.product(
            (1, 128), (False, True), (None, "token", "page"), (False, True)
        ):
            with self.subTest(
                ps=ps, mapped_full=mapped_full, swa=swa_kind, zero_tail=zero_tail
            ):
                self._check_tables(ps, mapped_full, swa_kind, zero_tail)

    def _check_tables(self, ps, mapped_full, swa_kind, zero_tail):
        device = "cuda"
        bs, width, delta, sentinel = 3, 300, 3, -17
        # Use strided inputs/outputs and mappings: graph replay must read the
        # original storage, without materializing a contiguous mapping copy.
        virtual_pages = torch.arange(bs * width, device=device).view(bs, width) % 31
        tokens = (
            virtual_pages[:, :, None] * ps + torch.arange(ps, device=device)
        ).reshape(bs, width * ps)
        req_to_token = torch.stack((tokens, tokens), dim=-1)[:, :, 0]
        rows = torch.tensor([1, 9, 2, 9, 0, 9], device=device)[::2]
        lengths = torch.tensor([2 * ps + 1, 1, 0], device=device, dtype=torch.int32)
        seq_lens = torch.stack((lengths - delta, lengths), dim=1)[:, 0]
        full = ((torch.arange(64, device=device, dtype=torch.int32) * 7) % 31)[::2]
        swa = ((torch.arange(64, device=device, dtype=torch.int32) * 11) % 31)[::2]
        full[21] = -1
        swa[22] = -1
        slots = torch.arange(32 * ps, device=device)
        token_map = ((slots // ps * 3 + 5) % 31) * ps + slots % ps
        token_map = torch.stack((token_map, token_map), dim=1)[:, 0]
        full_map = full if mapped_full else None
        swa_map = swa if swa_kind == "page" else None
        token_map = token_map if swa_kind == "token" else None
        full_storage = torch.full(
            (bs, 2 * (width + 2)), sentinel, device=device, dtype=torch.int32
        )
        swa_storage = torch.full_like(full_storage, sentinel)
        page_table = full_storage[:, ::2]
        swa_table = swa_storage[:, ::2] if swa_kind is not None else None
        cache_lens = torch.empty_like(lengths)
        cu_lens = torch.empty(bs + 1, device=device, dtype=torch.int32)

        normal_decode_set_metadata(
            cache_lens,
            cu_lens,
            page_table,
            req_to_token,
            rows,
            width,
            seq_lens,
            delta,
            ps,
            swa_page_table=swa_table,
            full_to_swa_index_mapping=token_map,
            full_v2p=full_map,
            swa_v2p=swa_map,
            zero_tail=zero_tail,
        )

        torch.testing.assert_close(cache_lens, lengths)
        torch.testing.assert_close(cu_lens[1:], lengths.cumsum(0).int())
        self.assertEqual(cu_lens[0].item(), 0)
        gathered = tokens[rows, ::ps]
        live = (
            torch.arange(width, device=device)[None, :]
            < ((lengths + ps - 1) // ps)[:, None]
        )
        expected_full = gathered // ps
        if mapped_full:
            expected_full = full[expected_full].clamp_min(0)
        expected_swa = None
        if swa_kind == "token":
            expected_swa = token_map[gathered] // ps
        elif swa_kind == "page":
            expected_swa = swa[gathered // ps].clamp_min(0)
        for storage, expected in (
            (full_storage, expected_full),
            (swa_storage, expected_swa),
        ):
            if expected is None:
                continue
            want = torch.full_like(storage, sentinel)
            want[:, : 2 * width : 2] = torch.where(
                live, expected, 0 if zero_tail else sentinel
            )
            torch.testing.assert_close(storage, want)

    def test_lengths_only_leaves_tables_untouched(self):
        for ps, skip in itertools.product((1, 128), (False, True)):
            with self.subTest(ps=ps, skip=skip):
                seq_lens = torch.tensor([0, 9, 130], device="cuda")
                cache_lens = torch.empty(3, device="cuda", dtype=torch.int32)
                cu_lens = torch.empty(4, device="cuda", dtype=torch.int32)
                table = torch.full((3, 4), 23, device="cuda", dtype=torch.int32)
                swa_table = table.clone()
                normal_decode_set_metadata(
                    cache_lens,
                    cu_lens,
                    table,
                    torch.zeros((3, 4 * ps), device="cuda", dtype=torch.int64),
                    torch.arange(3, device="cuda"),
                    4 if skip else 0,
                    seq_lens,
                    3,
                    ps,
                    swa_page_table=swa_table,
                    full_to_swa_index_mapping=torch.arange(4 * ps, device="cuda"),
                    skip_page_table=skip,
                    zero_tail=True,
                )
                torch.testing.assert_close(cache_lens, (seq_lens + 3).int())
                torch.testing.assert_close(
                    cu_lens,
                    torch.tensor([0, 3, 15, 148], device="cuda", dtype=torch.int32),
                )
                torch.testing.assert_close(table, torch.full_like(table, 23))
                torch.testing.assert_close(swa_table, table)


class _DraftLoad:
    """A DFLASH draft load: the slot copies (``pairs``, ``slot``) and the verify
    metadata, which reads its lengths and request rows from two of the slots."""

    DELTA, WIDTH = 16, 40

    def __init__(
        self, gen, *, bs, ps, rtt_dtype, scalar, embeds, from_slots, len_dtype
    ):
        dev, n = "cuda", bs * self.DELTA
        rows = bs + 9

        def rand(lo, hi, size, dtype=torch.int64):
            return torch.randint(lo, hi, size, generator=gen, device=dev).to(dtype)

        def slot_buffer(src):
            return torch.full((src.shape[0] + 5,), POISON, dtype=src.dtype, device=dev)

        # Past the table width too: the live page count clamps to it.
        self.seq_lens = rand(0, self.WIDTH * ps + 50, (bs,), len_dtype)
        self.req_pool_indices = torch.randperm(rows, generator=gen, device=dev)[:bs]
        self.req_to_token = rand(0, 1 << 24, (rows, self.WIDTH * ps + 3), rtt_dtype)
        # Strided like a batch field sliced out of a wider tensor.
        input_ids = rand(0, 1 << 30, (2 * n,))[::2]
        out_cache_loc = rand(1, 1 << 30, (n,), torch.int32)
        sources = [input_ids, out_cache_loc, self.seq_lens, self.req_pool_indices]
        self.buffers = [slot_buffer(src) for src in sources]
        self.pairs = [
            (buf[: src.shape[0]], src) for buf, src in zip(self.buffers, sources)
        ]
        self.lens_read, self.rows_read = self.pairs[2][0], self.pairs[3][0]
        if not from_slots:
            # The metadata reads tensors no copy writes.
            self.lens_read, self.rows_read = (
                self.seq_lens.clone(),
                self.req_pool_indices.clone(),
            )
            self.pairs = self.pairs[:2]
        if embeds:
            rows_src = torch.randn((n, 7168), generator=gen, device=dev).bfloat16()
            rows_buf = torch.full((n + 3, 7168), 7.0, dtype=torch.bfloat16, device=dev)
            self.buffers.append(rows_buf)
            words = (
                t.reshape(-1).view(torch.uint8).view(torch.int64)
                for t in (rows_buf[:n], rows_src)
            )
            self.pairs.append(tuple(words))
        self.slot = None
        if scalar is not None:
            count = torch.full((1,), POISON, dtype=torch.int32, device=dev)
            self.buffers.append(count)
            src = torch.full((), n - 3, dtype=torch.int32, device=dev)
            self.slot = (count, src, *scalar)
        self.cache_seqlens = torch.full(
            (bs + 3,), POISON, dtype=torch.int32, device=dev
        )
        self.cu_seqlens_k = torch.full((bs + 4,), POISON, dtype=torch.int32, device=dev)
        self.page_table = torch.full(
            (bs + 2, self.WIDTH + 7), POISON, dtype=torch.int32, device=dev
        )
        self.bs, self.ps = bs, ps

    def written(self):
        return [*self.buffers, self.cache_seqlens, self.cu_seqlens_k, self.page_table]

    def metadata_args(self, **overrides):
        args = dict(
            cache_seqlens_int32=self.cache_seqlens[: self.bs],
            cu_seqlens_k=self.cu_seqlens_k[: self.bs + 1],
            page_table=self.page_table[: self.bs],
            req_to_token=self.req_to_token,
            req_pool_indices=self.rows_read,
            max_seq_pages=self.WIDTH,
            seq_lens=self.lens_read,
            seq_len_delta=self.DELTA,
            page_size=self.ps,
        )
        return {**args, **overrides}

    def stock(self):
        fill_slots(self.pairs, self.slot)
        a = self.metadata_args()
        normal_decode_set_metadata(
            a["cache_seqlens_int32"],
            a["cu_seqlens_k"],
            a["page_table"],
            a["req_to_token"],
            a["req_pool_indices"],
            a["max_seq_pages"],
            a["seq_lens"],
            a["seq_len_delta"],
            a["page_size"],
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFillSlotsThenDecodeMetadata(CustomTestCase):
    """The one-launch draft load writes what the slot copies followed by the verify
    metadata write, reading the copied slots, write: every byte, nothing else."""

    def test_one_launch_writes_what_copies_then_metadata_write(self):
        i32, i64 = torch.int32, torch.int64
        # (bs, page size, table dtype, scalar slot write, embedding rows, metadata
        # reads the copied slots, length dtype); bs 64 with rows spans more copy
        # programs than page columns.
        cases = [
            (1, 1, i32, None, False, True, i64),
            (2, 16, i64, (1, 0, 0), True, True, i64),
            (5, 64, i32, (2, 5, 37), True, True, i32),
            (17, 16, i32, (2, 5, 37), True, False, i64),
            (64, 1, i64, (1, 0, 0), False, True, i64),
            (64, 16, i32, (2, 5, 37), True, True, i64),
            (130, 64, i32, (2, 5, 37), False, True, i64),
            (130, 16, i64, None, False, False, i32),
        ]
        for seed, case in enumerate(cases):
            bs, ps, rtt_dtype, scalar, embeds, from_slots, len_dtype = case
            kwargs = dict(
                bs=bs,
                ps=ps,
                rtt_dtype=rtt_dtype,
                scalar=scalar,
                embeds=embeds,
                from_slots=from_slots,
                len_dtype=len_dtype,
            )
            stock, fused = (
                _DraftLoad(torch.Generator("cuda").manual_seed(seed), **kwargs)
                for _ in range(2)
            )
            stock.stock()
            taken = fill_slots_then_decode_metadata(
                fused.pairs, fused.slot, **fused.metadata_args()
            )
            with self.subTest(**kwargs):
                self.assertTrue(taken)
                for want, got in zip(stock.written(), fused.written()):
                    self.assertTrue(torch.equal(want, got))

    def test_declines_write_nothing(self):
        """Each miss would read a slot before its copy lands, convert a value the
        metadata reads, write an operand another part reads, or take a layout the
        metadata kernel does not; the op then launches nothing."""
        gen = torch.Generator("cuda").manual_seed(3)
        load = _DraftLoad(
            gen,
            bs=4,
            ps=16,
            rtt_dtype=torch.int32,
            scalar=(1, 0, 0),
            embeds=False,
            from_slots=True,
            len_dtype=torch.int64,
        )
        lens_buf, rows_buf = load.buffers[2], load.buffers[3]
        loc_dst, loc_src = load.pairs[1]
        int32_lens = torch.zeros(9, dtype=torch.int32, device="cuda")
        converting = [*load.pairs[:2], (int32_lens[:4], load.seq_lens), load.pairs[3]]
        misses = {
            "no copy": dict(pairs=[]),
            "input partly a copy's destination": dict(seq_lens=lens_buf[1:5]),
            "input converted by its copy": dict(
                pairs=converting, seq_lens=int32_lens[:4]
            ),
            "input is the scalar slot": dict(seq_lens=load.slot[0].expand(4)),
            "output over a copy's destination": dict(cu_seqlens_k=loc_dst[:5]),
            "output over a copy's source": dict(cache_seqlens_int32=loc_src[:4]),
            "outputs overlap": dict(cu_seqlens_k=load.cache_seqlens[:5]),
            "table written by a copy": dict(
                pairs=[*load.pairs, (load.req_to_token[0, :4], rows_buf[:4])]
            ),
            "no page columns": dict(max_seq_pages=0),
            "page size not a power of two": dict(page_size=12),
            "int64 lengths out": dict(
                cache_seqlens_int32=torch.zeros(4, dtype=torch.int64, device="cuda")
            ),
            "short prefix sum": dict(cu_seqlens_k=load.cu_seqlens_k[:4]),
            "narrow page table": dict(page_table=load.page_table[:4, : load.WIDTH - 1]),
            "rows of another batch": dict(
                req_pool_indices=torch.arange(3, device="cuda")
            ),
            "host table": dict(req_to_token=load.req_to_token.cpu()),
        }
        for name, overrides in misses.items():
            pairs = overrides.pop("pairs", load.pairs)
            before = [t.clone() for t in load.written()]
            with self.subTest(miss=name):
                self.assertFalse(
                    fill_slots_then_decode_metadata(
                        pairs, load.slot, **load.metadata_args(**overrides)
                    )
                )
                for want, got in zip(before, load.written()):
                    self.assertTrue(torch.equal(want, got))


if __name__ == "__main__":
    unittest.main()
