from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_dflash_mamba_commit_kernel(
    prefix_lens,
    commit_lens,
    last_steps,
    track_steps,
    prefix_stride: tl.constexpr,
    commit_stride: tl.constexpr,
    batch_size: tl.constexpr,
    track_interval: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    valid = row < batch_size
    commit = tl.load(commit_lens + row * commit_stride, valid, 0).to(tl.int64)
    tl.store(last_steps + row, commit - 1, valid)
    if HAS_TRACK:
        prefix = tl.load(prefix_lens + row * prefix_stride, valid, 0).to(tl.int64)
        post = prefix + commit
        crossed = prefix // track_interval != post // track_interval
        step = tl.maximum(post // track_interval * track_interval - prefix - 1, 0)
        step = tl.where(crossed & (step < commit), step, -1)
        tl.store(track_steps + row, step, valid)


def _prepare_dflash_mamba_commit(
    prefix_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    last_steps_out: torch.Tensor,
    track_steps_out: torch.Tensor,
    track_interval: int,
    has_track: bool,
) -> None:
    bs = commit_lens.numel()
    if bs:
        _prepare_dflash_mamba_commit_kernel[(1,)](
            prefix_lens,
            commit_lens,
            last_steps_out,
            track_steps_out,
            prefix_lens.stride(0),
            commit_lens.stride(0),
            bs,
            track_interval,
            has_track,
            triton.next_power_of_2(bs),
            num_warps=1,
        )


@triton.jit
def _dflash_accept_bonus_contig_kernel(
    candidates_ptr,
    target_top1_ptr,
    accept_lens_out_ptr,
    commit_lens_out_ptr,
    bonus_ids_out_ptr,
    out_tokens_ptr,
    prefix_lens_ptr,
    new_seq_lens_out_ptr,
    candidates_row_stride,
    target_row_stride,
    accept_stride,
    commit_stride,
    bonus_stride,
    out_tokens_row_stride,
    prefix_lens_stride,
    new_seq_lens_stride,
    block_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size
    draft_mask = cols < (block_size - 1)

    candidate_row_ptr = candidates_ptr + row * candidates_row_stride
    target_row_ptr = target_top1_ptr + row * target_row_stride
    candidate_tail = tl.load(candidate_row_ptr + cols + 1, mask=draft_mask, other=0)

    accept_len = tl.full((), 0, tl.int32)
    prefix_live = tl.full((), 1, tl.int32)
    for col in range(BLOCK_SIZE - 1):
        in_range = col < (block_size - 1)
        candidate_id = tl.load(candidate_row_ptr + (col + 1), mask=in_range, other=0)
        target_id = tl.load(target_row_ptr + col, mask=in_range, other=0)
        match_i32 = (candidate_id == target_id).to(tl.int32)
        keep = in_range & (prefix_live != 0) & (match_i32 != 0)
        accept_len += keep.to(tl.int32)
        prefix_live = tl.where(in_range, prefix_live & match_i32, prefix_live)

    commit_len = accept_len + 1
    bonus_id = tl.load(target_row_ptr + accept_len.to(tl.int64))
    new_seq_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride) + commit_len

    tl.store(accept_lens_out_ptr + row * accept_stride, accept_len)
    tl.store(commit_lens_out_ptr + row * commit_stride, commit_len)
    tl.store(bonus_ids_out_ptr + row * bonus_stride, bonus_id)
    tl.store(new_seq_lens_out_ptr + row * new_seq_lens_stride, new_seq_len)

    out_val = tl.where(draft_mask, candidate_tail, 0)
    out_val = tl.where(cols == accept_len, bonus_id, out_val)
    tl.store(
        out_tokens_ptr + row * out_tokens_row_stride + cols, out_val, mask=row_mask
    )


def _pick_num_warps(block_size: int) -> int:
    if block_size <= 16:
        return 1
    if block_size <= 32:
        return 2
    if block_size <= 64:
        return 4
    return 8


def _is_row_major_contiguous_2d(x: torch.Tensor) -> bool:
    return x.ndim == 2 and x.is_contiguous()


def _compute_dflash_accept_bonus_triton_unchecked(
    candidates: torch.Tensor,
    target_top1: torch.Tensor,
    accept_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    bonus_ids_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    prefix_lens: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
) -> None:
    batch_size, block_size = candidates.shape
    if batch_size == 0:
        return

    if not _is_row_major_contiguous_2d(candidates):
        raise ValueError("DFLASH Triton accept_bonus requires contiguous candidates.")
    if not _is_row_major_contiguous_2d(target_top1):
        raise ValueError("DFLASH Triton accept_bonus requires contiguous target_top1.")
    if not _is_row_major_contiguous_2d(out_tokens_out):
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous out_tokens_out."
        )
    if not accept_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous accept_lens_out."
        )
    if not commit_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous commit_lens_out."
        )
    if not bonus_ids_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous bonus_ids_out."
        )
    if prefix_lens.ndim != 1:
        raise ValueError("DFLASH Triton accept_bonus requires 1D prefix_lens.")
    if not new_seq_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous new_seq_lens_out."
        )

    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _dflash_accept_bonus_contig_kernel[(batch_size,)](
        candidates,
        target_top1,
        accept_lens_out,
        commit_lens_out,
        bonus_ids_out,
        out_tokens_out,
        prefix_lens,
        new_seq_lens_out,
        candidates.stride(0),
        target_top1.stride(0),
        accept_lens_out.stride(0),
        commit_lens_out.stride(0),
        bonus_ids_out.stride(0),
        out_tokens_out.stride(0),
        prefix_lens.stride(0),
        new_seq_lens_out.stride(0),
        block_size,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit
def _prepare_dflash_draft_block_contig_kernel(
    bonus_tokens_ptr,
    prefix_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    block_ids_out_ptr,
    positions_out_ptr,
    cache_loc_out_ptr,
    bonus_tokens_stride,
    prefix_lens_stride,
    req_pool_indices_stride,
    req_to_token_row_stride,
    block_ids_row_stride,
    positions_row_stride,
    cache_loc_row_stride,
    req_to_token_width,
    block_size,
    mask_token_id,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size

    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride)
    bonus_token = tl.load(bonus_tokens_ptr + row * bonus_tokens_stride)

    logical_pos = prefix_len.to(tl.int64) + cols
    valid = row_mask & (logical_pos < req_to_token_width)
    req_row_ptr = req_to_token_ptr + req_idx * req_to_token_row_stride
    slot_ids = tl.load(req_row_ptr + logical_pos, mask=valid, other=0)

    block_ids = tl.full((BLOCK_SIZE,), mask_token_id, tl.int64)
    block_ids = tl.where(cols == 0, bonus_token.to(tl.int64), block_ids)
    tl.store(
        block_ids_out_ptr + row * block_ids_row_stride + cols, block_ids, mask=row_mask
    )
    tl.store(
        positions_out_ptr + row * positions_row_stride + cols,
        logical_pos,
        mask=row_mask,
    )
    tl.store(
        cache_loc_out_ptr + row * cache_loc_row_stride + cols,
        slot_ids.to(tl.int64),
        mask=row_mask,
    )


def _prepare_dflash_draft_block_unchecked(
    bonus_tokens: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    block_ids_out: torch.Tensor,
    positions_out: torch.Tensor,
    cache_loc_out: torch.Tensor,
    mask_token_id: int,
) -> None:
    batch_size = int(bonus_tokens.numel())
    if batch_size == 0:
        return

    if req_to_token.ndim != 2 or req_to_token.stride(1) != 1:
        raise ValueError("DFLASH Triton prepare_block requires row-major req_to_token.")
    if not _is_row_major_contiguous_2d(block_ids_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous block_ids_out."
        )
    if not _is_row_major_contiguous_2d(positions_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous positions_out."
        )
    if not _is_row_major_contiguous_2d(cache_loc_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous cache_loc_out."
        )

    block_size = int(block_ids_out.shape[1])
    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _prepare_dflash_draft_block_contig_kernel[(batch_size,)](
        bonus_tokens,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        block_ids_out,
        positions_out,
        cache_loc_out,
        bonus_tokens.stride(0),
        prefix_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        block_ids_out.stride(0),
        positions_out.stride(0),
        cache_loc_out.stride(0),
        int(req_to_token.shape[1]),
        block_size,
        int(mask_token_id),
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit(
    do_not_specialize=["n_rows", "block_size", "mask_token_id", "req_to_token_width"]
)
def _dflash_fork_prep_kernel(
    otb,
    otb_stride,
    fi,
    fi_stride,
    bonus_out,
    prefix_lens,
    prefix_lens_stride,
    req_pool_indices,
    req_pool_indices_stride,
    req_to_token,
    req_to_token_row_stride,
    block_ids_out,
    block_ids_row_stride,
    positions_out,
    positions_row_stride,
    cache_loc_out,
    cache_loc_row_stride,
    req_to_token_width,
    block_size,
    mask_token_id,
    n_rows,
    weight,
    emb_out,
    BLOCK_SIZE: tl.constexpr,
    EMB: tl.constexpr,
    hidden_dim: tl.constexpr,
    weight_stride0: tl.constexpr,
    org_vocab_start_index: tl.constexpr,
    org_vocab_end_index: tl.constexpr,
    num_org_vocab_padding: tl.constexpr,
    added_vocab_start_index: tl.constexpr,
    added_vocab_end_index: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Programs (r, c) with r < n_rows (= bs * block_size with EMB, else 0): embedding row r (request r // block_size,
    slot r % block_size), columns [c * BLOCK_H, ...). Programs (n_rows + b, 0): request b's bonus token and block row.
    """
    r = tl.program_id(0)
    c = tl.program_id(1)
    if r >= n_rows:
        row = r - n_rows
        if c != 0:
            return
        # The draft block kernel's body, its bonus token read from the relay row instead of a gathered tensor.
        cols = tl.arange(0, BLOCK_SIZE)
        row_mask = cols < block_size

        prefix_len = tl.load(prefix_lens + row * prefix_lens_stride)
        req_idx = tl.load(req_pool_indices + row * req_pool_indices_stride)
        bonus_token = tl.load(otb + tl.load(fi + row * fi_stride) * otb_stride)
        tl.store(bonus_out + row, bonus_token)

        logical_pos = prefix_len.to(tl.int64) + cols
        valid = row_mask & (logical_pos < req_to_token_width)
        req_row_ptr = req_to_token + req_idx * req_to_token_row_stride
        slot_ids = tl.load(req_row_ptr + logical_pos, mask=valid, other=0)

        block_ids = tl.full((BLOCK_SIZE,), mask_token_id, tl.int64)
        block_ids = tl.where(cols == 0, bonus_token.to(tl.int64), block_ids)
        tl.store(
            block_ids_out + row * block_ids_row_stride + cols, block_ids, mask=row_mask
        )
        tl.store(
            positions_out + row * positions_row_stride + cols,
            logical_pos,
            mask=row_mask,
        )
        tl.store(
            cache_loc_out + row * cache_loc_row_stride + cols,
            slot_ids.to(tl.int64),
            mask=row_mask,
        )
        return

    if EMB:
        # The embedding row gather's body, the token taken from the block row this launch writes.
        b = r // block_size
        j = r - b * block_size
        row64 = r.to(tl.int64)
        cols = c * BLOCK_H + tl.arange(0, BLOCK_H)
        col_mask = cols < hidden_dim

        bonus_token = tl.load(otb + tl.load(fi + b * fi_stride) * otb_stride)
        token = tl.where(
            j == 0, bonus_token.to(tl.int64), tl.full((), mask_token_id, tl.int64)
        )
        org_vocab_mask = (token >= org_vocab_start_index) & (
            token < org_vocab_end_index
        )
        added_vocab_mask = (token >= added_vocab_start_index) & (
            token < added_vocab_end_index
        )
        valid = org_vocab_mask | added_vocab_mask

        added_offset = (
            added_vocab_start_index
            - (org_vocab_end_index - org_vocab_start_index)
            - num_org_vocab_padding
        )
        local_id = tl.where(
            org_vocab_mask, token - org_vocab_start_index, token - added_offset
        )
        local_id = tl.where(valid, local_id, 0)
        vals = tl.load(
            weight + local_id * weight_stride0 + cols,
            mask=col_mask & valid,
            other=0.0,
        )
        tl.store(emb_out + row64 * hidden_dim + cols, vals, mask=col_mask)


def prepare_dflash_draft_block_from_relay(
    *,
    output_tokens: torch.Tensor,
    future_indices: torch.Tensor,
    bonus_tokens_out: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    block_ids_out: torch.Tensor,
    positions_out: torch.Tensor,
    cache_loc_out: torch.Tensor,
    mask_token_id: int,
    embedding_weight: Optional[torch.Tensor] = None,
    vocab_bounds: Optional[Tuple[int, int, int, int, int]] = None,
    embedding_out: Optional[torch.Tensor] = None,
) -> None:
    """``_prepare_dflash_draft_block_unchecked`` with its bonus tokens read from the
    overlap relay (``output_tokens[future_indices]``, also written to
    ``bonus_tokens_out``), in one launch.

    With ``embedding_weight``, the same launch writes ``embedding_out[b, j]`` = the
    table row of block id ``[b, j]`` as one vocabulary shard gathers it (``vocab_bounds``
    = org start, org end, org padding, added start, added end; ids outside both ranges
    read zeros). ``future_indices`` int32 / int64 ``[bs]``; ``bonus_tokens_out``
    contiguous ``[bs]``; the block outputs as the unfused kernel takes them;
    ``embedding_out`` contiguous ``[bs, block_size, hidden]`` of the table's dtype.
    """
    bs = int(bonus_tokens_out.numel())
    if bs == 0:
        return
    if req_to_token.ndim != 2 or req_to_token.stride(1) != 1:
        raise ValueError("DFLASH Triton prepare_block requires row-major req_to_token.")
    for name, out in (
        ("block_ids_out", block_ids_out),
        ("positions_out", positions_out),
        ("cache_loc_out", cache_loc_out),
    ):
        if not _is_row_major_contiguous_2d(out):
            raise ValueError(f"DFLASH Triton prepare_block requires contiguous {name}.")
    if not bonus_tokens_out.is_contiguous():
        raise ValueError("DFLASH fork prep requires contiguous bonus_tokens_out.")
    block_size = int(block_ids_out.shape[1])
    if embedding_weight is not None:
        if not (
            embedding_out is not None
            and embedding_out.is_contiguous()
            and embedding_weight.stride(1) == 1
            and tuple(embedding_out.shape)
            == (bs, block_size, int(embedding_weight.shape[1]))
        ):
            raise ValueError("DFLASH fork prep: embedding layout mismatch.")
        hidden = int(embedding_weight.shape[1])
        block_h = min(1024, triton.next_power_of_2(hidden))
        ncol = triton.cdiv(hidden, block_h)
        org_start, org_end, org_padding, added_start, added_end = vocab_bounds
        consts = dict(
            hidden_dim=hidden,
            weight_stride0=int(embedding_weight.stride(0)),
            org_vocab_start_index=int(org_start),
            org_vocab_end_index=int(org_end),
            num_org_vocab_padding=int(org_padding),
            added_vocab_start_index=int(added_start),
            added_vocab_end_index=int(added_end),
            BLOCK_H=block_h,
        )
        weight, out = embedding_weight, embedding_out
        rows = bs * block_size
    else:
        ncol = 1
        consts = dict(
            hidden_dim=1,
            weight_stride0=1,
            org_vocab_start_index=0,
            org_vocab_end_index=0,
            num_org_vocab_padding=0,
            added_vocab_start_index=0,
            added_vocab_end_index=0,
            BLOCK_H=16,
        )
        weight, out = block_ids_out, block_ids_out
        rows = 0
    _dflash_fork_prep_kernel[(rows + bs, ncol)](
        output_tokens,
        output_tokens.stride(0),
        future_indices,
        future_indices.stride(0),
        bonus_tokens_out,
        prefix_lens,
        prefix_lens.stride(0),
        req_pool_indices,
        req_pool_indices.stride(0),
        req_to_token,
        req_to_token.stride(0),
        block_ids_out,
        block_ids_out.stride(0),
        positions_out,
        positions_out.stride(0),
        cache_loc_out,
        cache_loc_out.stride(0),
        int(req_to_token.shape[1]),
        block_size,
        int(mask_token_id),
        rows,
        weight,
        out,
        BLOCK_SIZE=triton.next_power_of_2(block_size),
        EMB=1 if embedding_weight is not None else 0,
        num_warps=8,
        **consts,
    )


@triton.jit(do_not_specialize=["t_stride", "k_stride", "bs"])
def _stage_sampling_params_kernel(
    t_src, t_stride, t_dst, k_src, k_stride, g_dst, lo, bs, BLOCK: tl.constexpr
):
    """temperatures: t_dst[i] = lo if t[i] < lo else t[i] (NaN kept, as torch.clamp(min=lo)); greedy_mask: g_dst[i] =
    (top_ks[i] <= 1)."""
    i = tl.arange(0, BLOCK)
    m = i < bs
    x = tl.load(t_src + i * t_stride, mask=m)
    tl.store(t_dst + i, tl.where(x < lo, lo, x), mask=m)
    k = tl.load(k_src + i * k_stride, mask=m)
    tl.store(g_dst + i, k <= 1, mask=m)


def stage_sampling_params(
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    temperatures_out: torch.Tensor,
    greedy_mask_out: torch.Tensor,
    lo: float,
) -> None:
    """``temperatures_out = clamp(temperatures, min=lo)`` and ``greedy_mask_out =
    top_ks <= 1`` in one launch, bit-identical to the three torch ops.

    ``temperatures`` float32 and ``top_ks`` int32, ``[bs]`` with any stride;
    ``temperatures_out`` float32 and ``greedy_mask_out`` bool, contiguous ``[bs]``.
    """
    bs = int(temperatures_out.shape[0])
    _stage_sampling_params_kernel[(1,)](
        temperatures,
        int(temperatures.stride(0)),
        temperatures_out,
        top_ks,
        int(top_ks.stride(0)),
        greedy_mask_out,
        float(lo),
        bs,
        triton.next_power_of_2(max(bs, 16)),
    )


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    uniforms_ptr,
    temperatures_ptr,
    greedy_ptr,
    tokens_ptr,
    q_ptr,
    slots: tl.constexpr,
    top_k: tl.constexpr,
):
    """One program per request: a slot's K scores stay in registers and the walk is a
    loop, so the slot-to-slot dependency costs nothing instead of one kernel each."""
    row = tl.program_id(0)
    offsets = tl.arange(0, top_k)
    temperature = tl.load(temperatures_ptr + row)
    greedy = tl.load(greedy_ptr + row) != 0
    previous = 0
    for slot in range(slots):
        base = (row * slots + slot) * top_k
        scores = tl.load(scores_ptr + (base + previous) * top_k + offsets).to(
            tl.float32
        )
        if greedy:
            # Same pick as torch.argmax: the first NaN if any, else the first max.
            is_nan = scores != scores
            best = tl.max(tl.where(is_nan, -float("inf"), scores), axis=0)
            has_nan = tl.max(is_nan.to(tl.int32), axis=0) > 0
            hit = tl.where(has_nan, is_nan, scores == best)
            index = tl.min(tl.where(hit, offsets, top_k), axis=0)
            probabilities = tl.where(offsets == index, 1.0, 0.0)
        else:
            scaled = scores / temperature
            exponentials = tl.exp(scaled - tl.max(scaled, axis=0))
            probabilities = exponentials / tl.sum(exponentials, axis=0)
            uniform = tl.load(uniforms_ptr + row * slots + slot)
            index = tl.sum(
                tl.where(uniform >= tl.cumsum(probabilities, axis=0), 1, 0), axis=0
            )
            index = tl.minimum(index, top_k - 1)
        tl.store(q_ptr + base + offsets, probabilities)
        tl.store(tokens_ptr + row * slots + slot, tl.load(candidate_ptr + base + index))
        previous = index


def selector_walk_triton(
    *,
    candidate_ids,
    scores,
    uniforms,
    temperatures,
    greedy_mask,
):
    batch, slots, top_k = candidate_ids.shape
    tokens = torch.empty((batch, slots), dtype=torch.int64, device=scores.device)
    q_rows = torch.empty(
        (batch, slots, top_k), dtype=torch.float32, device=scores.device
    )
    _selector_walk_kernel[(batch,)](
        scores.contiguous(),
        candidate_ids.contiguous(),
        uniforms.contiguous(),
        temperatures.contiguous(),
        greedy_mask.contiguous(),
        tokens,
        q_rows,
        slots=slots,
        top_k=top_k,
        num_warps=1,
    )
    return tokens, q_rows


@triton.jit
def _fused_correct_drafts_and_bonus_kernel(
    candidates_ptr,
    target_predict_ptr,
    num_correct_drafts_ptr,
    bonus_tokens_ptr,
    block_size,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    in_row = offs < block_size - 1
    drafts = tl.load(candidates_ptr + b * block_size + 1 + offs, mask=in_row, other=-1)
    targets = tl.load(target_predict_ptr + b * block_size + offs, mask=in_row, other=-2)
    eq = (drafts == targets) & in_row
    # Leading-match count = index of the first mismatch lane; lanes past the
    # row and all-match rows both resolve to block_size - 1 via the min.
    num_correct = tl.min(tl.where(eq, BLOCK, offs), 0)
    bonus_token = tl.load(target_predict_ptr + b * block_size + num_correct)
    tl.store(num_correct_drafts_ptr + b, num_correct.to(tl.int32))
    tl.store(bonus_tokens_ptr + b, bonus_token.to(tl.int64))


def compute_dflash_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (greedy verify rule).

    Args:
        candidates: Token ids proposed by the DFlash draft, including the current token.
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
        target_predict: Token ids predicted by the target model for each position in the block.
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.

    Returns:
        correct_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token at index correct_len (the "bonus" token to append).

    Notes:
        Matches the reference implementation rule:
          accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    if candidates.is_cuda:
        num_correct_drafts = torch.empty(
            bs, dtype=torch.int32, device=candidates.device
        )
        bonus_tokens = torch.empty(bs, dtype=torch.int64, device=candidates.device)
        _fused_correct_drafts_and_bonus_kernel[(bs,)](
            candidates.contiguous(),
            target_predict.contiguous(),
            num_correct_drafts,
            bonus_tokens,
            block_size,
            BLOCK=triton.next_power_of_2(max(block_size - 1, 1)),
        )
        return num_correct_drafts, bonus_tokens

    matches = candidates[:, 1:] == target_predict[:, :-1]
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), correct_len]
    return correct_len.to(torch.int32), bonus.to(torch.int64)


@triton.jit
def _dflash_grouped_conv_kernel(
    hidden_ptr,
    delta_ptr,
    base_ptr,
    out_ptr,
    hidden_size,
    hidden_row_stride,
    delta_row_stride,
    delta_tap_stride,
    base_tap_stride,
    out_row_stride,
    block_size,
    TAPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden_size
    groups = cols // GROUP_SIZE
    position = row % block_size
    delta_row = delta_ptr + row * delta_row_stride + groups

    # Same fp32 expression as the compiled reference, rounded once at the store.
    x = tl.load(hidden_ptr + row * hidden_row_stride + cols, mask=mask)
    coef = tl.load(base_ptr + cols, mask=mask).to(tl.float32) + tl.load(
        delta_row, mask=mask
    ).to(tl.float32)
    out = coef * x.to(tl.float32)
    for tap in tl.static_range(1, TAPS):
        coef = tl.load(base_ptr + tap * base_tap_stride + cols, mask=mask).to(
            tl.float32
        ) + tl.load(delta_row + tap * delta_tap_stride, mask=mask).to(tl.float32)
        shifted = tl.load(
            hidden_ptr + (row - tap) * hidden_row_stride + cols,
            mask=mask & (position >= tap),
            other=0.0,
        )
        out = out + coef * shifted.to(tl.float32)
    tl.store(out_ptr + row * out_row_stride + cols, out, mask=mask)


def dflash_grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """Grouped dynamic depthwise conv of DFlashGroupedConv: hidden_states
    [tokens, hidden], delta [tokens, taps, hidden // group_size], base [taps,
    hidden]. Bit-exact with the torch.compile reference in models/dflash.py."""
    num_tokens, hidden_size = hidden_states.shape
    assert hidden_states.stride(1) == 1 and delta.stride(2) == 1
    assert base.stride(1) == 1 and delta.shape[1] == taps
    out = torch.empty_like(hidden_states)
    if num_tokens == 0:
        return out
    BLOCK = 1024
    grid = (num_tokens, triton.cdiv(hidden_size, BLOCK))
    _dflash_grouped_conv_kernel[grid](
        hidden_states,
        delta,
        base,
        out,
        hidden_size,
        hidden_states.stride(0),
        delta.stride(0),
        delta.stride(1),
        base.stride(0),
        out.stride(0),
        block_size,
        TAPS=taps,
        GROUP_SIZE=group_size,
        BLOCK=BLOCK,
        # GB300 sweep over 6..2304 tokens: 4 warps below a few hundred
        # tokens, 2 warps above (5-10% faster from 384 tokens).
        num_warps=4 if num_tokens < 256 else 2,
    )
    return out
