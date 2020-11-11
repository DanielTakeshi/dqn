from dqn.replay.experience import ExperienceReplay
from dqn.replay.episode import Episode
from dqn.utils.data_structures import SumSegmentTree, MinSegmentTree
import random
import numpy as np


class PrioritizedReplay(ExperienceReplay):
    def __init__(self, capacity, init_cap, frame_stack, gamma, alpha=0.6,
                 flag="replay"):
        super(PrioritizedReplay, self).__init__(
            capacity, init_cap, frame_stack, gamma, flag)
        assert alpha > 0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._flag = np.ones(capacity)

    def add_one_transition(self, transition):
        _pos = self._current_pos
        super().add_one_transition(transition)
        self._it_sum[_pos] = self._max_priority ** self._alpha
        self._it_min[_pos] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, self.__len__() - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def _register_one_episode(self, episode):
        assert isinstance(episode, Episode)
        for i in range(episode.length):
            _pos = self._current_pos
            self._register_frame(self.episode_number, i)
            self._it_sum[_pos] = self._max_priority ** self._alpha
            self._it_min[_pos] = self._max_priority ** self._alpha

    def sample(self, batch_size, num_steps, beta=0.4, p=None):
        """
        Sample a batch of experiences. It also returns importance weights and
        indexes of sampled experiences.
        """
        assert beta > 0
        assert type(num_steps) == list
        if self.__len__() < self.init_cap:
            return None
        keys = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.__len__()) ** (-beta)
        for key in keys:
            p_sample = self._it_sum[key] / self._it_sum.sum()
            weight = (p_sample * self.__len__()) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        return [self._encode_sample(keys, num_steps=n) for n in num_steps], \
               weights, keys

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions. Sets priority of transition at
        index idxes[i] in buffer to priorities[i].
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.__len__()
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def update_flags(self, start_idx, end_idx, flags):
        self._flag[start_idx: end_idx] = flags

    def get_flags(self, keys):
        return [self._flag[key] for key in keys]
