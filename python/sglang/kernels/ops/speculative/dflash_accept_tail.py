# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""The DFLASH accept tail: a decode step's new lengths and both overlap-relay rows in
one launch (``speculative/dflash_accept_tail.cuh``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_dflash_accept_tail_module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        "speculative_dflash_accept_tail",
        *args,
        cuda_files=["speculative/dflash_accept_tail.cuh"],
        cuda_wrappers=[("run", f"dflash_accept_tail::AcceptTail<{args}>::run")],
    )


@debug_kernel_api
def dflash_accept_tail(
    *,
    prefix_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    bonus: torch.Tensor,
    indices: torch.Tensor,
    new_seq_lens_buf: torch.Tensor,
    output_tokens_buf: torch.Tensor,
) -> torch.Tensor:
    """One launch: returns ``prefix_lens + commit_lens.to(int64)`` (a new tensor) and
    writes it to ``new_seq_lens_buf[indices]`` and ``bonus`` to
    ``output_tokens_buf[indices]``.

    ``prefix_lens``, ``bonus`` and ``indices`` int64, ``commit_lens`` int32, all
    contiguous ``[bs]`` on one CUDA device; both buffers contiguous int64. The
    ``indices`` must be distinct and in range of both buffers.
    """
    new_seq_lens = torch.empty_like(prefix_lens)
    _jit_dflash_accept_tail_module(is_arch_support_pdl()).run(
        prefix_lens,
        commit_lens,
        bonus,
        indices,
        new_seq_lens,
        new_seq_lens_buf,
        output_tokens_buf,
    )
    return new_seq_lens
