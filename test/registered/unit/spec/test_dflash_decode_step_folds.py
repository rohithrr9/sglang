"""DFLASH decode-step launch folds, against stub workers: a fold may drop launches,
never change a value the step reads."""

import unittest
from types import MethodType, SimpleNamespace

import torch

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


if __name__ == "__main__":
    unittest.main()
