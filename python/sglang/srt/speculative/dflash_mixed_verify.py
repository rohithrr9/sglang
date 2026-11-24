"""DFLASH mixed verify: the running batch's speculative step inside an extend pass.

When the scheduler runs an extend (prefill) pass while the running requests wait, the
running batch's next DFLASH step (its draft block and its target verify) runs inside
that pass: one target forward over the extend rows followed by one draft block per
running request. Each batch keeps its own result; the running batch's result is
processed first, as if its decode step had run before the extend pass.

Opt-in via ``SGLANG_OPT_DFLASH_MIXED_VERIFY``.
"""

from __future__ import annotations

import bisect
import collections
import copy
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class MixedVerifyLayout(msgspec.Struct, kw_only=True):
    """Row layout of a mixed verify pass, carried on the pass's ForwardBatch."""

    num_extend_reqs: int
    num_extend_tokens: int
    # New-token count of each extend request.
    extend_lens: list[int]
    num_verify_reqs: int
    # The verify step's CUDA-graph batch size for num_verify_reqs requests.
    num_padded_verify_reqs: int
    block_size: int
    # Logits pruning lengths on the device: extend_lens, then a 1 per verify row.
    prune_lens: torch.Tensor
    # Written during the forward: the attention split views (one build per pass)
    # and the verify rows' logits (this rank's vocab shard under a deferred gather).
    attn_views: Optional[Any] = None
    verify_logits: Optional[torch.Tensor] = None

    @property
    def num_verify_tokens(self) -> int:
        return self.block_size * self.num_verify_reqs


class DraftBlock(msgspec.Struct, kw_only=True, frozen=True):
    """One DFLASH draft block of a running batch, as the verify step consumes it."""

    bs: int
    block_size: int
    # [bs] committed lengths on the device.
    prefix_lens: torch.Tensor
    # [bs, block_size] and flat views of the verify positions and KV slots.
    positions_2d: torch.Tensor
    positions: torch.Tensor
    verify_out_cache_loc_2d: torch.Tensor
    verify_out_cache_loc: torch.Tensor
    iteration_ids: Any
    # [bs, block_size]: the bonus token, then the drafted tokens.
    draft_tokens: torch.Tensor
    ring_block_locs: Optional[torch.Tensor]


def pad_to_bucket(n: int, buckets: Sequence[int]) -> int:
    """The smallest bucket >= ``n``; ``n`` itself above the largest bucket."""
    i = bisect.bisect_left(buckets, n)
    return int(buckets[i]) if i < len(buckets) else int(n)


def pass_refusal(
    *, num_extend_tokens: int, num_verify_tokens: int, prefill_buckets: Sequence[int]
) -> Optional[str]:
    """Why a pass of these row counts must not mix, or None. A mixed pass rides the
    prefill CUDA graphs: within the largest bucket and padded at most 2x."""
    if not prefill_buckets:
        return None
    total = num_extend_tokens + num_verify_tokens
    if total > prefill_buckets[-1]:
        return "above the largest prefill bucket"
    if pad_to_bucket(total, prefill_buckets) > 2 * total:
        return "pass off the prefill graph"
    return None


def verify_prune_lens(
    extend_lens: Sequence[int], num_verify_reqs: int, block_size: int
) -> list[int]:
    """Pruning lengths that keep each extend row's last token and every verify token:
    each verify block is presented as ``block_size`` one-token rows."""
    return list(extend_lens) + [1] * (block_size * num_verify_reqs)


def prune_lens_to_device(
    prune_lens: Sequence[int], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    # Pinned staging and a non-blocking copy: the host keeps launching the pass.
    host = torch.tensor(list(prune_lens), dtype=dtype).pin_memory()
    return host.to(device, non_blocking=True)


def host_prefix_bounds(
    *, reserved_kv_lens: Optional[torch.Tensor], num_reqs: int, block_size: int
) -> tuple[Optional[list[int]], Optional[str]]:
    """``(bounds, None)`` or ``(None, reason)``. Host upper bounds of the running
    requests' committed lengths when the future map keeps them on the GPU only: the
    reserved next-KV lengths (the verify step's own host bound) minus the block."""
    if reserved_kv_lens is None or int(reserved_kv_lens.numel()) != num_reqs:
        return None, "no reserved lengths"
    bounds = [int(x) - block_size for x in reserved_kv_lens.tolist()]
    if min(bounds, default=0) < 0:
        return None, "reserved length below the block"
    return bounds, None


def build_mixed_batch(
    *,
    extend_batch: ScheduleBatch,
    running_batch: ScheduleBatch,
    draft_block: DraftBlock,
    verify_prefix_lens: Sequence[int],
) -> ScheduleBatch:
    """A shallow copy of ``extend_batch`` with one ``block_size``-token row per running
    request appended: its draft tokens at positions committed .. committed + block - 1,
    written to the request's reserved verify slots. ``verify_prefix_lens`` are the
    host prefix lengths (exact, or upper bounds; the device lengths are exact)."""
    E, R, block, bs = (
        extend_batch,
        running_batch,
        draft_block.block_size,
        draft_block.bs,
    )
    if len(verify_prefix_lens) != bs:
        raise RuntimeError(
            f"{len(verify_prefix_lens)} verify prefix lengths for {bs} running requests"
        )
    if R.req_pool_indices_cpu is None:
        raise RuntimeError(
            "Mixed verify needs the running batch's req_pool_indices_cpu"
        )
    verify_seq_lens = draft_block.prefix_lens + block
    m = copy.copy(E)
    m.reqs = list(E.reqs) + list(R.reqs)
    m.input_ids = torch.cat(
        [E.input_ids, draft_block.draft_tokens.reshape(-1).to(E.input_ids.dtype)]
    )
    m.req_pool_indices = torch.cat(
        [E.req_pool_indices, R.req_pool_indices.to(E.req_pool_indices.dtype)]
    )
    if E.req_pool_indices_cpu is not None:
        m.req_pool_indices_cpu = torch.cat(
            [
                E.req_pool_indices_cpu,
                R.req_pool_indices_cpu.to(E.req_pool_indices_cpu.dtype),
            ]
        )
    m.seq_lens = torch.cat([E.seq_lens, verify_seq_lens.to(E.seq_lens.dtype)])
    if E.seq_lens_cpu is not None:
        m.seq_lens_cpu = torch.cat(
            [
                E.seq_lens_cpu,
                torch.tensor(
                    [int(x) + block for x in verify_prefix_lens],
                    dtype=E.seq_lens_cpu.dtype,
                ),
            ]
        )
    if E.orig_seq_lens is not None:
        m.orig_seq_lens = torch.cat(
            [E.orig_seq_lens, verify_seq_lens.to(E.orig_seq_lens.dtype)]
        )
    m.seq_lens_sum = int(m.seq_lens_cpu.sum()) if m.seq_lens_cpu is not None else None
    m.out_cache_loc = torch.cat(
        [
            E.out_cache_loc,
            draft_block.verify_out_cache_loc.to(E.out_cache_loc.dtype),
        ]
    )
    m.prefix_lens = list(E.prefix_lens) + [int(x) for x in verify_prefix_lens]
    m.extend_lens = list(E.extend_lens) + [block] * bs
    m.extend_num_tokens = int(E.extend_num_tokens) + block * bs
    if E.extend_logprob_start_lens is not None:
        m.extend_logprob_start_lens = list(E.extend_logprob_start_lens) + [0] * bs
    if E.multimodal_inputs is not None:
        m.multimodal_inputs = list(E.multimodal_inputs) + [None] * bs
    if E.top_logprobs_nums is not None:
        m.top_logprobs_nums = list(E.top_logprobs_nums) + [0] * bs
    if E.token_ids_logprobs is not None:
        m.token_ids_logprobs = list(E.token_ids_logprobs) + [None] * bs
    m.return_logprob = False
    m.is_prefill_only = False
    return m


def fix_verify_positions(
    *,
    forward_batch: ForwardBatch,
    num_extend_reqs: int,
    verify_prefix_lens: torch.Tensor,
    compute_position: Callable,
    attention_backend_str: str,
) -> None:
    """Under host length bounds: the verify rows' device prefix lengths become the
    device-committed ones, and positions and extend starts are recomputed from them."""
    fb = forward_batch
    prefix = fb.extend_prefix_lens
    fb.extend_prefix_lens = torch.cat(
        [prefix[:num_extend_reqs], verify_prefix_lens.to(prefix.dtype)]
    )
    fb.positions, fb.extend_start_loc = compute_position(
        attention_backend_str,
        fb.extend_prefix_lens,
        fb.extend_seq_lens,
        fb.extend_num_tokens,
    )


# ----------------------------------------------------------------------- RNG order


def default_rng_generator() -> Optional[torch.Generator]:
    """The current device's default CUDA generator when it exposes its Philox offset."""
    if not torch.cuda.is_available():
        return None
    g = torch.cuda.default_generators[torch.cuda.current_device()]
    try:
        g.get_offset()
    except RuntimeError:
        return None
    return g


def sampling_key(sampling_info, num_reqs: int, vocab_size: int) -> tuple:
    """What the extend batch's sampling draws depend on."""
    if sampling_info is None:
        return (num_reqs, vocab_size, "none")
    return (
        num_reqs,
        vocab_size,
        bool(sampling_info.is_all_greedy),
        bool(sampling_info.need_top_p_sampling),
        bool(sampling_info.need_top_k_sampling),
        bool(sampling_info.need_min_p_sampling),
        sampling_info.sampling_seed is None,
    )


def measure_sampling_offset_step(
    *,
    sampler: Callable,
    sampling_info,
    num_reqs: int,
    vocab_size: int,
    device: torch.device,
    generator: torch.Generator,
    cache: dict,
) -> int:
    """Generator offset step of the extend batch's sampling: measured once per
    ``sampling_key`` by a ``sampler`` call on zero logits, offset restored after."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    key = sampling_key(sampling_info, num_reqs, vocab_size)
    step = cache.get(key)
    if step is not None:
        return step
    if sampling_info is None or sampling_info.is_all_greedy:
        step = 0
    else:
        logits = LogitsProcessorOutput(
            next_token_logits=torch.zeros(
                (num_reqs, vocab_size), dtype=torch.float32, device=device
            )
        )
        positions = torch.zeros((num_reqs,), dtype=torch.int64, device=device)
        offset = generator.get_offset()
        try:
            sampler(logits, sampling_info, False, None, None, positions)
            step = int(generator.get_offset() - offset)
        finally:
            generator.set_offset(offset)
    cache[key] = step
    return step


class RngOrder:
    """Serial draw order inside a mixed pass. The pass runs draft, forward, extend
    sampling, verify; the serial schedule draws extend sampling, draft, verify.
    ``generator`` None keeps the pass's own order."""

    def __init__(self, generator: Optional[torch.Generator], extend_step: int):
        self.generator, self.extend_step = generator, int(extend_step)
        self.start = generator.get_offset() if generator is not None else None
        self.end = None

    def before_draft(self) -> None:
        if self.generator is not None:
            self.generator.set_offset(self.start + self.extend_step)

    def before_extend_sampling(self) -> None:
        if self.generator is not None:
            self.end = self.generator.get_offset()
            self.generator.set_offset(self.start)

    def after_extend_sampling(self) -> bool:
        """Back to the offset after the draft's draws; False when the sampling's step
        differed from its measurement."""
        if self.generator is None:
            return True
        ok = self.generator.get_offset() == self.start + self.extend_step
        self.generator.set_offset(self.end)
        return ok


# ------------------------------------------------------------------ scheduler side


def running_batch_refusal(
    running_batch: Optional[ScheduleBatch],
    *,
    num_padded_reqs: int,
    max_verify_reqs: int,
) -> Optional[str]:
    """Why the running batch's next step must not fold, or None."""
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

    R = running_batch
    if R is None or R.is_empty():
        return "no running batch"
    if R.is_prefill_only:
        return "prefill-only running batch"
    if R.has_grammar:
        return "grammar"
    reason = _output_refusal(R)
    if reason is not None:
        return reason
    if not isinstance(R.spec_info, DFlashDraftInputV2):
        return "running spec input"
    if num_padded_reqs > max_verify_reqs:
        return "running batch above the verify request limit"
    return None


def extend_batch_refusal(
    extend_batch: Optional[ScheduleBatch],
    *,
    num_verify_tokens: int,
    prefill_buckets: Sequence[int],
) -> Optional[str]:
    """Why the extend batch must not carry the running batch's step, or None."""
    E = extend_batch
    if E is None or not E.forward_mode.is_extend() or E.forward_mode.is_mixed():
        return "not a plain extend"
    if E.is_prefill_only or E.has_grammar or E.return_logprob:
        return "extend flags"
    reason = _output_refusal(E)
    if reason is not None:
        return reason
    if E.input_embeds is not None:
        return "input embeds"
    if E.multimodal_inputs is not None and any(
        x is not None for x in E.multimodal_inputs
    ):
        return "multimodal"
    if (
        not E.extend_lens
        or E.prefix_lens is None
        or len(E.extend_lens) != len(E.prefix_lens)
    ):
        return "length lists"
    return pass_refusal(
        num_extend_tokens=int(sum(E.extend_lens)),
        num_verify_tokens=num_verify_tokens,
        prefill_buckets=prefill_buckets,
    )


def _output_refusal(batch: ScheduleBatch) -> Optional[str]:
    """Per-request outputs a mixed pass does not produce."""
    if any(req.return_routed_experts for req in batch.reqs):
        return "routed experts"
    if any(req.beam_group is not None for req in batch.reqs):
        return "beam search"
    sampling_info = batch.sampling_info
    if (
        sampling_info is not None
        and sampling_info.sampling_mask_batch_indices is not None
    ):
        return "sampling mask"
    return None


class DFlashMixedVerify:
    """Scheduler collaborator: reserves the verify rows in the prefill budget, folds
    the running batch's step into an extend pass, and hands the fold to run_batch."""

    def __init__(
        self,
        *,
        block_size: int,
        prefill_buckets: Sequence[int],
        decode_buckets: Sequence[int],
        max_verify_reqs: int,
    ):
        self.block_size = int(block_size)
        self.prefill_buckets = sorted(int(x) for x in prefill_buckets)
        self.decode_buckets = sorted(int(x) for x in decode_buckets)
        self.max_verify_reqs = int(max_verify_reqs)
        # Generator offset step of the extend sampling, per sampling_key.
        self.rng_steps: dict = {}
        self.refusals: collections.Counter = collections.Counter()
        self.fold_ct = 0
        self._pending: Optional[tuple] = None

    def num_padded_verify_reqs(self, num_reqs: int) -> int:
        """The batch size the verify step runs at: its CUDA graph's padded size."""
        return pad_to_bucket(num_reqs, self.decode_buckets)

    def reserved_tokens(self, running_batch: Optional[ScheduleBatch]) -> int:
        """Prefill-budget tokens held back for the running batch's verify rows."""
        reason = running_batch_refusal(
            running_batch,
            num_padded_reqs=self._padded_reqs_of(running_batch),
            max_verify_reqs=self.max_verify_reqs,
        )
        if reason is not None:
            return 0
        return self.block_size * len(running_batch.reqs)

    def maybe_fold(
        self,
        *,
        extend_batch: Optional[ScheduleBatch],
        running_batch: ScheduleBatch,
        prepare_decode: Callable[[ScheduleBatch], Optional[ScheduleBatch]],
    ) -> None:
        """Fold the running batch's next step into ``extend_batch`` when both qualify;
        ``prepare_decode`` is the scheduler's own decode preparation of the batch."""
        self._pending = None
        if extend_batch is None or not extend_batch.forward_mode.is_extend():
            return
        reason = running_batch_refusal(
            running_batch,
            num_padded_reqs=self._padded_reqs_of(running_batch),
            max_verify_reqs=self.max_verify_reqs,
        ) or extend_batch_refusal(
            extend_batch,
            num_verify_tokens=self.block_size * len(running_batch.reqs),
            prefill_buckets=self.prefill_buckets,
        )
        if reason is not None:
            self._refuse(reason)
            return
        decode_batch = prepare_decode(running_batch)
        if decode_batch is None or decode_batch.is_empty():
            self._refuse("running batch skipped its decode step")
            return
        self._pending = (extend_batch, decode_batch)
        self.fold_ct += 1

    def take_fold(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """The running batch folded into ``batch``, once; None for any other batch."""
        pending, self._pending = self._pending, None
        if pending is None or pending[0] is not batch:
            return None
        return pending[1]

    def _padded_reqs_of(self, running_batch: Optional[ScheduleBatch]) -> int:
        if running_batch is None:
            return 0
        return self.num_padded_verify_reqs(len(running_batch.reqs))

    def _refuse(self, reason: str) -> None:
        if reason == "no running batch":
            return
        if reason not in self.refusals:
            logger.info("DFLASH mixed verify: an extend pass runs unmixed (%s)", reason)
        self.refusals[reason] += 1


def mixed_verify_server_refusal(
    *,
    enable_overlap: bool,
    require_mlp_sync: bool,
    enable_hisparse: bool,
    has_dllm: bool,
    is_disaggregated: bool,
    pp_size: int,
    enable_encoder_swa_bounded_replay: bool,
    enable_unified_memory: bool,
    has_uocr_runtime: bool,
    has_confidence_budget: bool,
) -> Optional[str]:
    """Why this server's scheduling cannot run mixed verify passes, or None."""
    if not enable_overlap:
        return "overlap scheduler off"
    if require_mlp_sync:
        return "DP attention"
    if enable_hisparse or has_dllm:
        return "hisparse / dllm"
    if is_disaggregated:
        return "disaggregation"
    if pp_size != 1:
        return "pipeline parallelism"
    if enable_encoder_swa_bounded_replay:
        return "encoder SWA replay"
    if enable_unified_memory:
        return "unified memory"
    if has_uocr_runtime:
        return "uocr runtime"
    if has_confidence_budget:
        return "confidence budget"
    return None


def build_dflash_mixed_verify(
    *, model_worker, target_model_runner, server_refusal: Optional[str]
) -> Optional[DFlashMixedVerify]:
    """The scheduler collaborator, or None (logged) when this server cannot mix."""
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
    from sglang.srt.utils import is_hip

    reason = server_refusal
    if reason is None and not isinstance(model_worker, DFlashWorkerV2):
        reason = "not the DFLASH v2 worker"
    if reason is None:
        reason = model_worker.mixed_verify_refusal()
    backend = target_model_runner.attn_backend
    if reason is None and (not isinstance(backend, TRTLLMHAAttnBackend) or is_hip()):
        reason = "attention backend other than TRT-LLM MHA"
    if reason is None and backend.use_fmha_v2:
        reason = "fmha_v2 prefill"
    if reason is not None:
        logger.warning(
            "SGLANG_OPT_DFLASH_MIXED_VERIFY is set, but extend passes run unmixed: %s",
            reason,
        )
        return None
    prefill_runner = target_model_runner.prefill_cuda_graph_runner
    decode_runner = target_model_runner.decode_cuda_graph_runner
    return DFlashMixedVerify(
        block_size=int(model_worker.block_size),
        prefill_buckets=(
            prefill_runner.capture_num_tokens
            if isinstance(prefill_runner, PrefillCudaGraphRunner)
            else []
        ),
        decode_buckets=(
            decode_runner.capture_bs
            if isinstance(decode_runner, DecodeCudaGraphRunner)
            and target_model_runner.decode_cuda_graph_captured
            else []
        ),
        # Requests per verify step the verify kernels are sized for at boot.
        max_verify_reqs=int(target_model_runner.max_running_requests) + 1,
    )
