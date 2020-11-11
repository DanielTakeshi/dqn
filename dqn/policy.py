import numpy as np
from dqn.utils.schedules import PiecewiseSchedule
import logging


class GreedyEpsilonPolicy(object):

    def __init__(self, params, num_actions, max_num_steps, train_freq_per_step):
        self.params = params
        self.num_actions = num_actions
        self.max_num_steps = max_num_steps
        self.train_freq_per_step = train_freq_per_step
        # ----------------------------------------------------------------------
        # We'll use frames_mid and frames_end, and default to going from 1 to
        # 0.1 for first 1M frames, then decreasing to 0.01 at 4M frames and
        # staying there. This is NOT the blend scheudle -- don't be confused.
        # ----------------------------------------------------------------------
        assert train_freq_per_step in [1,2,4], \
                "We shouldn't be testing settings other than these."
        self.schedule = PiecewiseSchedule([
            (0,                    params["start"]),
            (params["frames_mid"], params["mid"]),
            (params['frames_end'], params["end"])
        ], outside_value=params["end"])
        self.selector = lambda x: x.argmax()
        logger = logging.getLogger("greedy-eps")
        logger.debug("Just defined policy.")
        logger.debug("  params[start]: {}".format(params["start"]))
        logger.debug("  params[mid]:   {} @ {}".format(params["mid"], params['frames_mid']))
        logger.debug("  params[end]:   {} @ {}".format(params["end"], params['frames_end']))

    def __call__(self, qs, steps=None):
        assert isinstance(qs, np.ndarray)
        if steps and np.random.uniform() <= self.get_epsilon(steps):
            return np.random.randint(0, self.num_actions)
        actions = self.selector(qs)
        return actions

    def get_epsilon(self, steps):
        return self.schedule.value(steps)
