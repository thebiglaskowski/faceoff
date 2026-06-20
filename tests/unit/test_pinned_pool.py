"""Tests for pinned frame pools."""

import numpy as np


class TestPinnedFramePool:
    def test_borrow_reuses_buffer(self):
        from utils.pinned_pool import PinnedFramePool

        pool = PinnedFramePool(8, 8, capacity=2)
        a = pool.borrow(0)
        b = pool.borrow(0)
        c = pool.borrow(1)
        assert a is b
        assert a is not c
        assert a.shape == (8, 8, 3)
        assert a.dtype == np.uint8