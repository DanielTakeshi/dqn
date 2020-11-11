import numpy as np


class Transition(object):

    def __init__(self, state, next_state, action, reward, done, weight=1.0):
        """A tuple that stores a simple transition in the environment.

        This can be somewhat confusing, though. We use this both to represent a
        single transition (in which case, reward, done, etc., are single values)
        and a batch of transitions, in which case these are lists (or numpy
        arrays).
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        if done and state is not None:
            next_state = state
        self.next_state = next_state
        self.weight = weight


    def stack(self):
        """
        If transition is a list of samples, it will convert the list into a
        `numpy` array.
        """
        if isinstance(self.state, list):
            self.state = np.array(self.state, copy=False)
        if isinstance(self.next_state, list):
            self.next_state = np.array(self.next_state, copy=False)
        if isinstance(self.action, list):
            self.action = np.array(self.action)
        if isinstance(self.reward, list):
            self.reward = np.array(self.reward, dtype=np.float32)
        if isinstance(self.done, list):
            self.done = np.array(self.done, dtype=np.float32)
        if isinstance(self.weight, list):
            self.weight = np.array(self.weight, dtype=np.float32)


    def __len__(self):
        if isinstance(self.action, np.ndarray):
            return len(self.action)
        else:
            return 1


    def get_transitions(self, idx, num_steps):
        assert isinstance(self.action, np.ndarray)
        _transitions = Transition(state=[], next_state=[], action=[], reward=[],
                                  done=[])
        _transitions.state      = self.state[idx]
        _transitions.next_state = self.next_state[idx]
        _transitions.action     = self.action[idx]
        _transitions.reward     = self.reward[idx]
        _transitions.done       = self.done[idx]
        _transitions.weight     = self.weight[idx]
        return _transitions
