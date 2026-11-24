"""The one-launch CUDA graph slot fill: every byte it writes must be what fill_from's
grouped copies and post-fill hooks write, and nothing else."""

import itertools
import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.memory.slot_fill import fill_slots
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    FillContext,
    build_decode_registry,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")

POISON = 0x3C3C3C3C


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFillSlots(CustomTestCase):
    def test_copies_and_scalar_match_copy_and_clamp(self):
        gen = torch.Generator("cuda").manual_seed(0)
        dtype_pairs = list(itertools.product((torch.int32, torch.int64), repeat=2))
        # Block size flips at 64 elements; int64 values past int32 wrap like copy_.
        for n, scalar in itertools.product(
            (1, 63, 64, 65, 300), (None, (1, 0, 0), (2, 3, 5), (2, 0, 1 << 20))
        ):
            pairs, wants, backings = [], [], []
            for i, (dst_dtype, src_dtype) in enumerate(dtype_pairs):
                size = max(1, n - 7 * i)
                src = torch.randint(
                    -(1 << 40), 1 << 40, (2 * size,), generator=gen, device="cuda"
                ).to(src_dtype)[::2]
                backing = torch.full(
                    (size + 3,), POISON, dtype=dst_dtype, device="cuda"
                )
                pairs.append((backing[:size], src))
                wants.append(torch.empty_like(backing[:size]).copy_(src))
                backings.append(backing)
            slot, slot_want = None, None
            if scalar is not None:
                mode, offset, cap = scalar
                value = torch.randint(-4, 12, (1,), generator=gen, device="cuda")
                value = value.to(torch.int32)
                slot_buf = torch.full((1,), POISON, dtype=torch.int32, device="cuda")
                slot = (slot_buf, value.reshape(()), mode, offset, cap)
                slot_want = value if mode == 1 else torch.clamp(value - offset, 0, cap)
            fill_slots(pairs, slot)
            with self.subTest(n=n, scalar=scalar):
                for (dst, _), want, backing in zip(pairs, wants, backings):
                    self.assertTrue(torch.equal(dst, want))
                    self.assertTrue(torch.all(backing[dst.numel() :] == POISON))
                if slot is not None:
                    self.assertTrue(torch.equal(slot[0], slot_want))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFusedRegistryFill(CustomTestCase):
    TOKENS_PER_REQ = 8

    def _registry(self, *, fuse, cache_loc_dtype=torch.int64, sharded=False, **kwargs):
        registry = build_decode_registry(
            device=torch.device("cuda"),
            max_bs=64,
            max_num_token=64 * self.TOKENS_PER_REQ,
            seq_len_fill_value=1,
            cache_loc_dtype=cache_loc_dtype,
            enable_num_token_non_padded=True,
            attn_tp_sharded_fn=lambda _n: sharded,
            share_pool=False,
            fuse_fills=fuse,
            **kwargs,
        )
        for name in registry.slot_names():
            registry.get_slot(name).buffer.fill_(POISON % 1000003)
        return registry

    def _batch(self, gen, bs, form):
        n = bs * self.TOKENS_PER_REQ
        seq_lens = torch.randint(100_000, 250_000, (bs,), generator=gen, device="cuda")
        # Below the token count: the sharded target clamps inside its range.
        count = torch.full((), n - 3, dtype=torch.int32, device="cuda")
        return SimpleNamespace(
            input_ids=torch.randint(0, 202048, (n,), generator=gen, device="cuda"),
            positions=torch.randint(
                0, 250_000, (bs, self.TOKENS_PER_REQ), generator=gen, device="cuda"
            ).reshape(-1),
            out_cache_loc=torch.randint(1, 1 << 30, (n,), generator=gen, device="cuda"),
            req_pool_indices=torch.randperm(4096, generator=gen, device="cuda")[:bs],
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            mrope_positions=None,
            # The draft carries its local count, the target verify the global one.
            num_token_non_padded=count if form == "draft" else None,
            global_num_token_non_padded=None if form == "draft" else count,
        )

    def test_fused_fill_writes_what_fill_from_writes(self):
        gen = torch.Generator("cuda").manual_seed(0)
        for cache_loc_dtype, form, sharded, bs in itertools.product(
            (torch.int64, torch.int32),
            ("draft", "target"),
            (False, True),
            (1, 2, 3, 4, 5, 8, 64),
        ):
            n = bs * self.TOKENS_PER_REQ
            batch = self._batch(gen, bs, form)
            unpadded = dict(
                raw_bs=bs, padded_bs=bs, raw_num_tokens=n, padded_num_tokens=n
            )
            padded_bs = 64 if bs < 64 else bs
            padded = dict(unpadded, padded_bs=padded_bs)
            padded["padded_num_tokens"] = padded_bs * self.TOKENS_PER_REQ
            for shape, skip in (
                (unpadded, frozenset()),
                (unpadded, frozenset({"input_ids", "positions"})),
                (padded, frozenset()),
            ):
                with get_parallel().override(attn_tp_size=2, attn_tp_rank=1):
                    stock, fused = (
                        self._registry(
                            fuse=fuse, cache_loc_dtype=cache_loc_dtype, sharded=sharded
                        )
                        for fuse in (False, True)
                    )
                    engaged = (
                        fused._plan_fused_fill(batch, FillContext(**shape), skip)
                        is not None
                    )
                    stock.fill_from(batch, skip_slots=skip, **shape)
                    fused.fill_from(batch, skip_slots=skip, **shape)
                with self.subTest(
                    cache_loc=cache_loc_dtype,
                    form=form,
                    sharded=sharded,
                    bs=bs,
                    padded=shape is padded,
                    skip=sorted(skip),
                ):
                    self.assertEqual(engaged, shape is unpadded or bs == 64)
                    for name in stock.slot_names():
                        self.assertTrue(
                            torch.equal(
                                stock.get_slot(name).buffer, fused.get_slot(name).buffer
                            ),
                            name,
                        )

    def test_recorded_fill_writes_device_slots_only_at_its_launch(self):
        """A recorded fill writes its host slots at once (fill_from is not called
        after it) and its device slots only when launched, then as fill_from does."""
        gen = torch.Generator("cuda").manual_seed(2)
        batch = self._batch(gen, 4, "draft")
        shape = dict(raw_bs=4, padded_bs=4, raw_num_tokens=32, padded_num_tokens=32)
        stock, held = self._registry(fuse=False), self._registry(fuse=False)
        stock.fill_from(batch, **shape)
        recorded = held.record_fill(batch, **shape)
        self.assertIsNotNone(recorded)
        for name in held.slot_names():
            buf = held.get_slot(name).buffer
            want = stock.get_slot(name).buffer if buf.device.type == "cpu" else None
            with self.subTest(name=name, launched=False):
                if want is None:
                    self.assertTrue(torch.all(buf == POISON % 1000003))
                else:
                    self.assertTrue(torch.equal(buf, want))
        recorded.launch()
        for name in held.slot_names():
            with self.subTest(name=name, launched=True):
                self.assertTrue(
                    torch.equal(held.get_slot(name).buffer, stock.get_slot(name).buffer)
                )
        untouched = self._registry(fuse=False)
        padded = dict(shape, padded_bs=8, padded_num_tokens=64)
        self.assertIsNone(untouched.record_fill(batch, **padded))
        for name in untouched.slot_names():
            buf = untouched.get_slot(name).buffer
            self.assertTrue(torch.all(buf == POISON % 1000003), name)

    def test_steps_aside_where_one_launch_differs(self):
        """Each miss would otherwise skip a write fill_from makes (padding resets, a
        side input), write a value differently (a gathered hook, a converted scalar, a
        source sharing its slot's storage) or address a field it cannot."""
        gen = torch.Generator("cuda").manual_seed(1)
        registry = self._registry(fuse=True)
        batch = self._batch(gen, 4, "target")
        shape = dict(raw_bs=4, padded_bs=4, raw_num_tokens=32, padded_num_tokens=32)

        def plan(*, target=registry, skip=frozenset(), ctx=None, **fields):
            fb = SimpleNamespace(**{**vars(batch), **fields})
            return target._plan_fused_fill(
                fb, FillContext(**{**shape, **(ctx or {})}), skip
            )

        self.assertIsNotNone(plan())
        device_slots = ("input_ids", "positions", "out_cache_loc", "req_pool_indices")
        misses = {
            "padded batch": dict(ctx=dict(padded_bs=8, padded_num_tokens=64)),
            "padded tokens": dict(ctx=dict(padded_num_tokens=40)),
            "pipeline proxy": dict(ctx=dict(pp_proxy_tensors=object())),
            "sliced slot": dict(
                mrope_positions=torch.zeros((3, 32), dtype=torch.int64, device="cuda")
            ),
            "gathered hook": dict(
                target=self._registry(fuse=True, require_gathered_buffer=True)
            ),
            "int64 token count": dict(
                global_num_token_non_padded=torch.tensor([29], device="cuda")
            ),
            "host source": dict(seq_lens=batch.seq_lens.cpu()),
            "float source": dict(positions=batch.positions.float()),
            "source in its slot's storage": dict(
                input_ids=registry.get_slot("input_ids").buffer[32:64]
            ),
            "no device copy": dict(skip=frozenset(device_slots + ("seq_lens",))),
        }
        for name, overrides in misses.items():
            with self.subTest(miss=name):
                self.assertIsNone(plan(**overrides))


if __name__ == "__main__":
    unittest.main()
