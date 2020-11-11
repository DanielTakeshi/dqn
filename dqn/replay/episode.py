import numpy as np
from dqn.replay.transition import Transition
from dqn.utils.train import compute_discount
# Causing issues with tkinter if computer doesn't have it
# (It's not pip-installable)
#try:
#    import matplotlib.pyplot as plt
#    import matplotlib.animation as animation
#except ImportError:
#    print("Couldn't install tkinter -- not really important")
import os


class Episode(object):

    def __init__(self, episode_num, init_obs):
        """A memory efficient way to store an episode.

        The (84,84) states are not duplicated, so a given (4,84,84) must take a
        slice out of `self.states`. Also, for 'transition' objects, unless we're
        at the first one in an episoe, they only save a 'next state', not the
        current one.

        :param episode_num: The index of the current episode, ONE-INDEXED.
        :param init_obs: Initial observation of the current episode
        """
        self.length = 0
        self.states = []
        self.rewards = []
        self.actions = []
        self.episode_num = episode_num
        self.episode_total_reward = 0.0
        self.done = False
        self.current_obs = init_obs
        self._init_obs(init_obs)

    @property
    def rew(self):
        """Returns latest single transition rewards, CLIPPED.
        """
        assert len(self.rewards) > 0
        return self.rewards[-1]

    def _init_obs(self, states):
        """Add initialize observations to the episode one by one.

        :param states: A numpy array of initial states, with first index of
        total number of frames to be added
        """
        assert states[0].shape == (84,84)
        for i in range(states.shape[0]):
            self.states.append(states[i])

    def add_transition(self, transition):
        """Add a new transition to the current episode memory.

        When we initialize a new episode, self.length=1 but len(self.states)=5
        because we save s_0 and s_1, so we basically have (5,84,84) data.

        :param transition: A `Transition` object. Under this memory efficient
            implementation, it holds None for the field `state`.
        """
        assert not self.done
        assert isinstance(transition, Transition)
        assert transition.state is None
        if transition.next_state.shape[0] == 1:
            self.current_obs = None
            self.states.append(transition.next_state[0])
        else:
            self.current_obs = transition.next_state
            self.states.append(transition.next_state[-1])
        self.actions.append(transition.action)
        self.rewards.append(transition.reward)
        self.length += 1
        self.episode_total_reward += transition.reward
        self.done = transition.done
        if transition.done:
            self._finish_episode()

    def sample_transition(self, idx, frame_stack, num_steps, gamma, force):
        """Retrieve information corresponding to episode index we drew.

        We're not actually 'sampling', we already have the indices that we
        randomly drew. We're just getting the information for that.

        Elements in `self.states` are shape (84,84). So you _need_ to include
        `frame_stack=4` --- as we do by default --- to match prior work. And by
        default, `num_steps=1` so if we have the frames: (f1,f2,f3,f4,f5,...),
        where each is (84,84), then states and next state are (f1,f2,f3,f4) and
        (f2,f3,f4,f5), respectively. Should be the same as what OpenAI does. And
        `done`s are also handled simply by how this is a self-contained episode!

        The reward should be the (num_steps)-step returns, EXCEPT, obviously,
        for the last part which requires the boot-strapped Q-network. EXCEPT,
        when we have the next state corresponding to a done state, in which
        case we don't use the bootstrapped estimate. This is the same as I
        implemented it in baselines, so long as `force=False`, so we do NOT
        return the last frame. So far I've always seen it False, good...

        We deal with only ONE 'lifespan' in this episode object, so it's not a
        full episode except for games that don't have lives, like Pong. It's
        still fine because if we lost a life, done=True (even if episode might
        not be done) so we'll ignore the `_next_state`. And in the next episode
        object (i.e., next 'lifespan') we correctly do an `env.reset()`.

        :param idx: Starting index of the transition which should in between
            0 and self.length - 1
        :param frame_stack: Number of frames to stack in the samples
        :param num_steps: Number of frame gap between states and next_states
        :param gamma: Discount rate
        :param force: if `force=True`, we always return the last frame even
            if episode frames are not enough
        :return: A tuple of the sample (state, next_state, action, reward)
        """
        assert 0 <= idx < self.length
        assert num_steps >= 1
        assert frame_stack >= 1
        _state_start_idx = idx
        _state_end_idx = idx + frame_stack
        _next_state_start_idx = _state_start_idx + num_steps
        _next_state_end_idx = _state_end_idx + num_steps
        _state = self.states[_state_start_idx:_state_end_idx]
        _rews = 0
        _done = False
        if _next_state_end_idx >= self.length:
            if force:
                _next_state = self.states[-frame_stack:]
            else:
                _next_state = None
            _done = True
        else:
            _next_state = self.states[_next_state_start_idx:_next_state_end_idx]
            _rews = compute_discount(self.rewards[idx:idx+num_steps], gamma)
        return Transition(state=_state, next_state=_next_state,
                          action=self.actions[idx], reward=_rews,
                          done=_done)

    def _finish_episode(self):
        """
        Things to do when an episode is finished. Here, all transitions are
        stacked into numpy arrays.
        """
        assert self.done
        self.states = np.stack(self.states)
        self.actions = np.stack(self.actions)
        self.rewards = np.stack(self.rewards)

    def to_animation(self, dir_episodes, flag="train"):
        """Generate a GIF animation of the current episode.

        :param dir_episodes: path of the directory to save the GIF
        :param flag: flag in the saved file, usually between train and test
        """
        assert self.done
        fig, ax = plt.subplots(1, figsize=(5, 5))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ims = []
        for im in self.states:
            _im = ax.imshow(np.rot90(np.flip(im, 1)),
                            alpha=0.5,
                            extent=(0, 1, 1, 0),
                            animated=True)
            ax.axis("off")
            ax.axis("tight")
            ims.append([_im])
        ani = animation.ArtistAnimation(fig, ims, blit=False, interval=1)
        writer = animation.ImageMagickWriter()
        ani.save(
            os.path.join(dir_episodes, "episode_{0}_{1}.gif".format(
                flag, str(self.episode_num).zfill(7))), writer=writer)
        plt.close("all")
