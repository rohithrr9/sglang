"""DFLASH decode-step launch folds, against stub workers: a fold may drop launches,
never change a value the step reads."""

import unittest
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.managers.overlap_utils import FutureRelayBuffers
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _bind(worker, *names):
    for name in names:
        setattr(worker, name, MethodType(getattr(DFlashWorkerV2, name), worker))
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


if __name__ == "__main__":
    unittest.main()
