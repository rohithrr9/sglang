# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""A CUDA graph slot fill in one launch: the grouped device copies of the batch fields
into their slots and one int32 scalar slot write."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

# (destination, source, mode, offset, cap) of the scalar slot write: mode 1 copies the
# int32 scalar, mode 2 writes clamp(source - offset, 0, cap).
ScalarSlotWrite = Tuple[torch.Tensor, torch.Tensor, int, int, int]


@triton.jit(
    do_not_specialize=["sizes", "strides", "slot_off", "slot_cap"],
    do_not_specialize_on_alignment=["sizes", "strides", "slot_off", "slot_cap"],
)
def _fill_slots_kernel(
    srcs,
    dsts,
    sizes,
    strides,
    NF: tl.constexpr,
    slot_src,
    slot_dst,
    slot_off,
    slot_cap,
    SLOT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Program (f, b): field f < NF copies elements [b * BLOCK, (b + 1) * BLOCK) of srcs[f] (element stride strides[f])
    to the contiguous dsts[f], converted to the destination's dtype. Program (NF, 0) writes the slot: SLOT 1 = copy of
    the int32 scalar; SLOT 2 = clamp(scalar - slot_off, 0, slot_cap) in int32."""
    f = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    for i in tl.static_range(NF):
        if f == i:
            m = offs < sizes[i]
            v = tl.load(srcs[i] + offs.to(tl.int64) * strides[i], mask=m)
            tl.store(dsts[i] + offs, v.to(dsts[i].dtype.element_ty), mask=m)
    if SLOT != 0:
        if f == NF:
            if tl.program_id(1) == 0:
                g = tl.load(slot_src)
                if SLOT == 2:
                    g = g - slot_off
                    g = tl.minimum(tl.maximum(g, 0), slot_cap)
                tl.store(slot_dst, g)


def fill_slots(
    pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    slot: Optional[ScalarSlotWrite] = None,
) -> None:
    """``dst.copy_(src)`` for each (dst, src) of ``pairs`` and the ``slot`` write, in one
    launch with the values ``Tensor.copy_`` and ``torch.clamp`` give.

    ``pairs``: at least one; 1-D int32 / int64 CUDA tensors of equal length, ``dst``
    contiguous; ``slot``: int32 one-element ``dst`` and ``src`` on the same device.
    """
    dsts = tuple(d for d, _ in pairs)
    srcs = tuple(s for _, s in pairs)
    sizes = tuple(int(d.numel()) for d in dsts)
    strides = tuple(int(s.stride(0)) for s in srcs)
    largest = max(sizes) if sizes else 1
    block = 64 if largest <= 64 else 256
    mode, s_dst, s_src, off, cap = 0, None, None, 0, 0
    if slot is not None:
        s_dst, s_src, mode, off, cap = slot
    nf = len(pairs)
    anchor = (dsts or (s_dst,))[0]
    _fill_slots_kernel[(nf + (1 if mode else 0), triton.cdiv(largest, block))](
        srcs,
        dsts,
        sizes,
        strides,
        nf,
        s_src if s_src is not None else anchor,
        s_dst if s_dst is not None else anchor,
        int(off),
        int(cap),
        mode,
        block,
    )
