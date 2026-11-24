"""Attention of a DFLASH mixed verify pass (see speculative/dflash_mixed_verify.py).

Every attention call of the pass is split in two. The extend rows run on an EXTEND
view of the pass. The verify rows run on the TARGET_VERIFY view the served verify
step builds for the same requests: cache length = committed + block, ``block_size``
queries per request, padded to the verify CUDA graph's batch size (padding rows: one
token before the block, request slot 0, write slot 0). Only TRT-LLM MHA metadata is
split.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional, Sequence

import msgspec
import torch

from sglang.srt.model_executor.forward_mode import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.attention.trtllm_mha_backend import (
        TRTLLMHAAttnBackend,
        TRTLLMMHAMetadata,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.dflash_mixed_verify import MixedVerifyLayout


class MixedVerifyViews(msgspec.Struct, kw_only=True, frozen=True):
    """The two views of one pass, keyed by the pass's metadata and ForwardBatch."""

    metadata: TRTLLMMHAMetadata
    forward_batch: ForwardBatch
    extend_metadata: TRTLLMMHAMetadata
    extend_batch: ForwardBatch
    verify_metadata: TRTLLMMHAMetadata
    verify_batch: ForwardBatch


def forward_mixed_verify(
    backend: TRTLLMHAAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache: bool = True,
    **kwargs,
) -> torch.Tensor:
    """One attention call of a mixed verify pass: extend rows, then verify rows."""
    layout = forward_batch.mixed_verify
    views = layout.attn_views
    metadata = backend.forward_metadata
    if (
        views is None
        or views.metadata is not metadata
        or views.forward_batch is not forward_batch
    ):
        views = layout.attn_views = build_views(
            backend=backend, forward_batch=forward_batch, layout=layout
        )
    num_extend, num_verify = layout.num_extend_tokens, layout.num_verify_tokens
    rows = q.shape[0]
    if rows < num_extend + num_verify or k is None or v is None:
        raise RuntimeError(
            f"Mixed verify attention got {rows} rows (k/v {k is not None}/"
            f"{v is not None}) for a pass of {num_extend} + {num_verify} rows"
        )
    out = forward_batch._attn_output
    if out is None:
        out = torch.empty(q.shape, dtype=backend.q_data_type, device=q.device)
    flat = out.reshape(rows, -1)
    extend_batch = views.extend_batch
    extend_batch._attn_output = flat[:num_extend]
    verify = slice(num_extend, num_extend + num_verify)
    pad = layout.block_size * layout.num_padded_verify_reqs - num_verify
    try:
        backend.forward_metadata = views.extend_metadata
        out_extend = backend.forward_extend(
            q[:num_extend],
            k[:num_extend],
            v[:num_extend],
            layer,
            extend_batch,
            save_kv_cache,
            **kwargs,
        )
        backend.forward_metadata = views.verify_metadata
        out_verify = forward_verify_rows(
            backend,
            pad_rows(q[verify], pad),
            pad_rows(k[verify], pad),
            pad_rows(v[verify], pad),
            layer,
            views.verify_batch,
            **kwargs,
        )
    finally:
        backend.forward_metadata = metadata
        extend_batch._attn_output = None
    if out_extend.data_ptr() != flat.data_ptr():
        flat[:num_extend].copy_(out_extend.reshape(num_extend, -1))
    flat[verify].copy_(out_verify.reshape(-1, flat.shape[1])[:num_verify])
    return flat


def forward_verify_rows(
    backend: TRTLLMHAAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    **kwargs,
) -> torch.Tensor:
    """The verify rows' attention call (padded rows; the verify view is set on the
    backend). An extension point: plugins replace the call per layer."""
    return backend.forward_extend(q, k, v, layer, forward_batch, True, **kwargs)


def pad_rows(t: Optional[torch.Tensor], n: int) -> Optional[torch.Tensor]:
    """``t`` with ``n`` zero rows appended (``t`` itself when ``n`` is 0)."""
    if t is None or n <= 0:
        return t
    return torch.cat([t, t.new_zeros((n,) + tuple(t.shape[1:]))])


def decode_tail_split(extend_lens: Sequence[int]) -> tuple[int, int]:
    """TRT-LLM MHA's split of a plain extend before trailing one-token requests:
    (prefix requests, prefix tokens); (-1, 0) when there is no one-token tail."""
    prefix = len(extend_lens)
    while prefix and extend_lens[prefix - 1] == 1:
        prefix -= 1
    if prefix < len(extend_lens):
        return prefix, int(sum(extend_lens[:prefix]))
    return -1, 0


def build_views(
    *,
    backend: TRTLLMHAAttnBackend,
    forward_batch: ForwardBatch,
    layout: MixedVerifyLayout,
) -> MixedVerifyViews:
    metadata = backend.forward_metadata
    for name in (
        "fmha_v2_copy_rows",
        "fmha_v2_copy_cols",
        "fmha_v2_block_table",
        "zigzag_page_table",
        "zigzag_swa_page_table",
        "encoder_cache_seqlens",
    ):
        if getattr(metadata, name) is not None:
            raise RuntimeError(f"Mixed verify cannot split metadata with {name} set")
    fill = int(backend.get_cuda_graph_seq_len_fill_value())
    return MixedVerifyViews(
        metadata=metadata,
        forward_batch=forward_batch,
        extend_metadata=_extend_metadata(
            metadata, layout, plain_extend=forward_batch.spec_info is None
        ),
        extend_batch=_extend_batch(forward_batch, layout),
        verify_metadata=_verify_metadata(metadata, layout, fill=fill),
        verify_batch=_verify_batch(forward_batch, metadata, layout, fill=fill),
    )


def _extend_metadata(
    md: TRTLLMMHAMetadata, layout: MixedVerifyLayout, *, plain_extend: bool
) -> TRTLLMMHAMetadata:
    n, tokens = layout.num_extend_reqs, layout.num_extend_tokens
    view = copy.copy(md)
    view.cache_seqlens_int32 = md.cache_seqlens_int32[:n]
    view.cu_seqlens_q = md.cu_seqlens_q[: n + 1]
    view.cu_seqlens_k = md.cu_seqlens_k[: n + 1]
    view.max_seq_len_q = int(max(layout.extend_lens))
    view.page_table = md.page_table[:n]
    if md.swa_page_table is not None:
        view.swa_page_table = md.swa_page_table[:n]
    if md.swa_out_cache_loc is not None:
        view.swa_out_cache_loc = md.swa_out_cache_loc[:tokens]
    view.mixed_prefill_reqs, view.mixed_prefill_tokens = (
        decode_tail_split(layout.extend_lens) if plain_extend else (-1, 0)
    )
    return view


def _extend_batch(fb: ForwardBatch, layout: MixedVerifyLayout) -> ForwardBatch:
    n, tokens = layout.num_extend_reqs, layout.num_extend_tokens
    view = copy.copy(fb)
    view.mixed_verify = None
    view.batch_size = n
    view.out_cache_loc = fb.out_cache_loc[:tokens]
    for name in (
        "seq_lens",
        "req_pool_indices",
        "extend_seq_lens",
        "extend_prefix_lens",
        "extend_start_loc",
        "seq_lens_cpu",
        "extend_seq_lens_cpu",
        "extend_prefix_lens_cpu",
        "extend_logprob_start_lens_cpu",
    ):
        value = getattr(fb, name)
        if value is not None:
            setattr(view, name, value[:n])
    if fb.positions is not None:
        view.positions = fb.positions[:tokens]
    return view


def _verify_metadata(
    md: TRTLLMMHAMetadata, layout: MixedVerifyLayout, *, fill: int
) -> TRTLLMMHAMetadata:
    n, bs, block = layout.num_extend_reqs, layout.num_verify_reqs, layout.block_size
    num_pad_reqs = layout.num_padded_verify_reqs - bs
    num_tokens = block * bs
    num_padded_tokens = block * layout.num_padded_verify_reqs
    cache_seqlens = md.cache_seqlens_int32[n : n + bs]
    if num_pad_reqs:
        cache_seqlens = torch.cat(
            [
                cache_seqlens,
                torch.full(
                    (num_pad_reqs,),
                    fill + block,
                    dtype=cache_seqlens.dtype,
                    device=cache_seqlens.device,
                ),
            ]
        )
    # A fresh metadata object: every field the split does not set keeps its default.
    view = type(md)()
    view.cache_seqlens_int32 = cache_seqlens.contiguous()
    view.max_seq_len_q = block
    view.cu_seqlens_q = torch.arange(
        0,
        num_padded_tokens + 1,
        block,
        dtype=torch.int32,
        device=cache_seqlens.device,
    )
    view.cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(view.cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
    )
    view.page_table = pad_rows(md.page_table[n : n + bs], num_pad_reqs).contiguous()
    if md.swa_page_table is not None:
        view.swa_page_table = pad_rows(
            md.swa_page_table[n : n + bs], num_pad_reqs
        ).contiguous()
    if md.swa_out_cache_loc is not None:
        start = layout.num_extend_tokens
        view.swa_out_cache_loc = pad_rows(
            md.swa_out_cache_loc[start : start + num_tokens],
            num_padded_tokens - num_tokens,
        )
    view.is_ragged_verify = False
    return view


def _verify_batch(
    fb: ForwardBatch, md: TRTLLMMHAMetadata, layout: MixedVerifyLayout, *, fill: int
) -> ForwardBatch:
    n, bs, block = layout.num_extend_reqs, layout.num_verify_reqs, layout.block_size
    num_pad_reqs = layout.num_padded_verify_reqs - bs
    start, num_tokens = layout.num_extend_tokens, block * bs
    num_pad_tokens = block * num_pad_reqs
    view = copy.copy(fb)
    view.mixed_verify = None
    view.forward_mode = ForwardMode.TARGET_VERIFY
    view.batch_size = layout.num_padded_verify_reqs
    view.out_cache_loc = pad_rows(
        fb.out_cache_loc[start : start + num_tokens], num_pad_tokens
    )
    seq_lens = md.cache_seqlens_int32[n : n + bs].to(fb.seq_lens.dtype) - block
    if num_pad_reqs:
        seq_lens = torch.cat(
            [
                seq_lens,
                torch.full(
                    (num_pad_reqs,), fill, dtype=seq_lens.dtype, device=seq_lens.device
                ),
            ]
        )
    view.seq_lens = seq_lens
    view.req_pool_indices = pad_rows(fb.req_pool_indices[n : n + bs], num_pad_reqs)
    if fb.positions is not None:
        view.positions = pad_rows(
            fb.positions[start : start + num_tokens], num_pad_tokens
        )
    for name in (
        "seq_lens_cpu",
        "extend_seq_lens",
        "extend_prefix_lens",
        "extend_start_loc",
        "extend_seq_lens_cpu",
        "extend_prefix_lens_cpu",
        "extend_logprob_start_lens_cpu",
    ):
        setattr(view, name, None)
    view.spec_info = None
    view._attn_output = None
    return view
