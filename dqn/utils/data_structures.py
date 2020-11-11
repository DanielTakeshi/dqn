import numpy as np
from collections import deque
import operator


class SMAQueue:
    def __init__(self, size):
        self.queue = deque()
        self.size = size

    def __iadd__(self, other):
        if isinstance(other, (list, tuple)):
            self.queue.extend(other)
        else:
            self.queue.append(other)
        while len(self.queue) > self.size:
            self.queue.popleft()
        return self

    def __len__(self):
        return len(self.queue)

    def __repr__(self):
        return "SMAQueue(size=%d)" % self.size

    def __str__(self):
        return "SMAQueue(size=%d, len=%d)" % (self.size, len(self.queue))

    def min(self):
        if not self.queue:
            return None
        return np.min(self.queue)

    def mean(self):
        if not self.queue:
            return None
        return np.mean(self.queue)

    def max(self):
        if not self.queue:
            return None
        return np.max(self.queue)


class Tracker:
    def __init__(self, logger, stop_reward=None, max_num_steps=None):
        self.stop_reward = stop_reward
        self.max_num_steps = max_num_steps
        self.logger = logger
        self.mean_reward = None

    def __enter__(self):
        self.total_episodes = 0
        self.mean_rewards = None
        return self

    def __exit__(self, *args):
        pass

    def reward(self, mean_rewards, frame):
        """D: used for our processes to determine whether we should exit.  Two
        cases: if we hit the max designated reward, or the max frames.
        """
        self.total_episodes += 1
        self.mean_reward = mean_rewards
        if self.max_num_steps and frame >= self.max_num_steps:
            self.logger.info(
                "Maximum {0} number of steps reached with {1} episodes"
                .format(self.max_num_steps, self.total_episodes))
            return True
        if self.stop_reward and self.mean_reward > self.stop_reward:
            self.logger.info(
                "Target reward {0} has been reached with {1} frames"
                .format(self.stop_reward, frame))
            return True
        return False


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient `reduce`
               operation which reduces `operation` over
               a contiguous subsequence of items in the
               array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must for a mathematical group together with the set of
            possible values for array elements.
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)
