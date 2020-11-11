import numpy as np
import logging
import random
from dqn.replay.episode import Episode
from dqn.replay.transition import Transition
from dqn.agent.agent import Agent
from dqn.environment.setup import Environment
from dqn.utils.math import relative_perplexity
# Don't need unless plotting, since it causes tkinter problems
#from dqn.utils.plot import generate_top_transitions_grid, plot_tsne
from dqn.utils.io import write_dict
from dqn.utils.experience import merge_transitions_xp
import torch
import os
import torchvision.utils as vutils
from pprint import pformat
from scipy.misc import imsave


class ExperienceReplay(object):
    def __init__(self, writer, capacity, init_cap, frame_stack, gamma, tag,
                 debug_dir=None):
        """
        A regular experience replay buffer. The buffer is maintained by
        holding `Episode` object for each episode. When the size of the
        replay buffer is met, it sets the first episode to `None`.

        :param writer: TensorBoard writer for debugging purposes
        :param capacity: The total amount of transitions to hold in the buffer
        :param init_cap: The initial amount of transitions in the buffer in
        order to call `sample` function
        :param frame_stack: Number of frames to stack in the samples
        :param gamma: Discount rate
        :param tag: tag for logging purposes
        :param debug_dir: an output directory for storing debugging output
        """
        self.writer = writer
        self._buffer = []
        self.capacity = capacity
        self._debug_dir = debug_dir
        self._tag = tag
        # Total number of frames covered by all active episodes in the memory
        self.total_active_trans = 0
        # Table storing the mapping of replay index to a tuple of
        # (episode_number, frame_number)
        self._frame_lookup = {}
        # Table storing the mapping of episode number to the index in `_buffer`
        self._episode_lookup = {}
        self._current_pos = 0
        self._init_cap = init_cap
        self._frame_stack = frame_stack
        self._gamma = gamma
        # The most recent episode
        self._current_episode = None
        self._first_active_idx = 0
        self.info = {}
        self._debug_output = {}
        # Number of frames in the last episode not included in sampling
        self.logger = logging.getLogger(tag + "_replay")
        if debug_dir and not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        self.logger.debug("Replay buffer with capacity {0} has been "
                          "initialized".format(capacity))

    @property
    def buffer(self):
        return self._buffer

    @property
    def total_episodes(self):
        """All episodes that have EVER been added to the replay buffer.

        Including episodes that are evicted, which are assigned to `None` in
        this list, in the order they are evicted, so this list always starts
        with some number of `None` items (including potentially none, which
        happens before we over-ride any samples).
        """
        return len(self._buffer)

    @property
    def total_active_episodes(self):
        """
        :return only the active episodes that have ever been added to the
        replay buffer.
        """
        return self.total_episodes - self._first_active_idx

    def __len__(self):
        """
        :return: the true replay buffer size. The replay buffer contains
        episodes or transitions more than its capacity, but it only for explicit
        storage purposes. The `__len__` function outputs the true size,
        which is always bounded by the capacity.
        """
        return len(self._frame_lookup)

    def _convert_episode_idx(self, episode_num):
        assert episode_num in self._episode_lookup
        return self._buffer[self._episode_lookup[episode_num]]

    def _get_transition(self, i, num_steps, force=False):
        """Find a sample inside the replay buffer.

        :param i: the index of the transition in the replay buffer lookup
        table
        :param num_steps: Number of steps to look forward
        :param force: if `force=True`, we always return the last frame even
        if episode frames are not enough
        :return: A `Transition` object
        """
        assert 0 <= i < self.__len__()
        _episode_num, _frame_num = self._frame_lookup[i]
        assert _episode_num in self._episode_lookup
        assert self._episode_lookup[_episode_num] >= self._first_active_idx
        return self._convert_episode_idx(_episode_num).sample_transition(
            idx=_frame_num, frame_stack=self._frame_stack,
            num_steps=num_steps, gamma=self._gamma, force=force)

    def get_transitions(self, idx, num_steps, force=False):
        """
        Find multiple samples in the replay buffer and output a numpy array
        stacking all samples together.

        :param idx: An iterable that contains indexes of transitions
        :param num_steps: Number of steps to look forward
        :param force: if `force=True`, we always return the last frame even
        if episode frames are not enough
        :return: A `Transition` object, where each field is a `numpy` array
        stacked all requested samples
        """
        assert num_steps >= 1
        _transitions = []
        for i in idx:
            _transitions.append(self._get_transition(i, num_steps, force))
        return merge_transitions_xp(_transitions)

    def sample_one(self, num_steps, output_idx=False):
        if self.__len__() < self._init_cap:
            return None
        idx = self._sample_idx(1)
        if output_idx:
            return self._get_transition(i=idx[0], num_steps=num_steps), idx[0]
        else:
            return self._get_transition(i=idx[0], num_steps=num_steps)

    def sample(self, batch_size, num_steps, output_idx=False):
        """Called from `agent.dqn.sample_transitions()`.

        Also from `SnapshotsTeacher.get_teacher_samples()` for teachers.

        Sample in the replay buffer. If not enough transitions (true size less
        than `init_cap`, return `None`, but `atari.py` guards against this.

        :param batch_size: the size of the batch to be sampled
        :param num_steps: A list of number of steps to look forward
        :param output_idx: A boolean controlling if index also outputs
        along with the samples
        :return: A numpy array of `Transition` object, with index if
        `output_idx` is True
        """
        if self.__len__() < self._init_cap:
            return None
        idx = self._sample_idx(batch_size)
        if output_idx:
            return self.get_transitions(idx=idx, num_steps=num_steps), idx
        else:
            return self.get_transitions(idx=idx, num_steps=num_steps)

    def _sample_idx(self, bs):
        assert self.__len__() >= self._init_cap
        if "p" in self.info:
            idx = np.random.choice(self.__len__(), bs, replace=False, p=self.info["p"])
        else:
            # https://github.com/CannyLab/dqn/issues/28
            #idx = np.random.choice(self.__len__(), bs, replace=False)
            idx = random.sample( range(self.__len__()) , bs)
        return idx

    def add_episode(self, episode):
        """Add a LIFESPAN to replay buffer and evict older ones if necessary.

        :param episode: An `Episode` object
        """
        assert isinstance(episode, Episode)
        self._register_one_episode(next_episode_idx=self.total_episodes, episode=episode)
        self._add_one_episode(episode)
        self._evict_episodes()

    def _register_frame(self, episode_number, frame_number):
        """
        Register a frame in the `_frame_lookup` table for quick selection
        later on. Once a frame is registered, it moves the counter
        `_current_pos` by 1. The frame number idx will increase faster than
        the episode number, since there are many frames per episode.

        :param episode_number: current episode number
        :param frame_number: current frame number
        """
        self._frame_lookup[self._current_pos] = (episode_number, frame_number)
        self._current_pos = (self._current_pos + 1) % self.capacity

    def _register_one_episode(self, next_episode_idx, episode):
        """Register frames for a lifespan.

        :param next_episode_idx: Index in `_buffer` for the current episode
        :param episode: An `Episode` object
        """
        assert isinstance(episode, Episode)
        self._episode_lookup[episode.episode_num] = next_episode_idx
        for i in range(episode.length):
            self._register_frame(episode.episode_num, i)

    def _add_one_episode(self, episode):
        """Add episode to the replay buffer and write debug message.

        Called internally and the _shared_ XP replay subclass. Finally
        `self._buffer` contains the memory of the lifespan.

        :param episode: An `Episode` object
        """
        assert isinstance(episode, Episode)
        self._buffer.append(episode)
        self.total_active_trans += episode.length
        self.logger.debug("Life {0} (1-idx) of length {1} added into replay, "
            "with {2}/{3} active lifespans, {4} total frames and {5} "
            "registered frames.".format(
                episode.episode_num, episode.length,
                self.total_active_episodes, self.total_episodes,
                self.total_active_trans, self.__len__()))

    def _evict_episodes(self):
        """Evict early episodes if capacity is met.

        The capacity is a soft upper bound. We can exceed it, and the while
        condition means evict episodes so long as the outcome _after_ eviction
        means we have at least the capacity. By design, we therefore do not go
        _under_ the capacity after we've filled up the buffer initially.  I
        think this is by design: the `frame_lookup` depends on there being
        something at all indices, and if we were to go below capacity, we would
        have some 'dead/broken' indices (again, after filling buffer).

        'Evicting' here means setting the corresponding buffer item to None.
        """
        while self.total_active_trans - self.capacity >= \
                self._buffer[self._first_active_idx].length:
            _old_episode_num = self._buffer[self._first_active_idx].episode_num
            _old_episode_length = self._buffer[self._first_active_idx].length
            self.total_active_trans -= _old_episode_length
            self._buffer[self._first_active_idx] = None
            self._first_active_idx += 1
            self.logger.debug(
                "Life {0} of length {1} was evicted from replay memory, "
                "with {2}/{3} active lifespans, {4} total frames and {5} "
                "registered frames.".format(
                    _old_episode_num, _old_episode_length,
                    self.total_active_episodes, self.total_episodes,
                    self.total_active_trans, self.__len__()))

    def _init_episode(self, init_obs):
        """
        Maintain the replay buffer by adding transitions. The buffer is
        stored in terms of episodes so that before adding new transitions,
        it is necessary to initialize a new `Episode` to hold experience.

        :param init_obs: Initial observations of the episodes
        """
        assert self._current_episode is None
        assert init_obs.shape[0] == self._frame_stack
        self._current_episode = Episode(
            episode_num=self.total_episodes + 1,  # One-Index!
            init_obs=init_obs)
        self.add_episode(self._current_episode)

    def add_one_transition(self, transition):
        """
        Maintain the replay buffer by adding one transition to the current
        `Episode`. Called during (at least) normal D-DQN training. The states
        are 8-bit integers, for efficient image storage.

        :param transition: A `Transition` object
        """
        assert isinstance(transition, Transition)
        assert transition.next_state.shape[0] == 1
        if self._current_episode is None:
            # ------------------------------------------------------------------
            # From Allen: for the first transition in the episode, the states
            # consists of both current states and next_states. All other
            # transition only needs to contain the new frame from `next_state`.
            # ------------------------------------------------------------------
            assert transition.state.shape[0] == self._frame_stack
            self._init_episode(transition.state)
            transition.state = None
        assert transition.state is None
        self._register_frame(self._current_episode.episode_num,
                             self._current_episode.length)
        self._current_episode.add_transition(transition)
        self.total_active_trans += 1
        if transition.done:
            self.logger.debug("Life {0} finished w/{1} transition "
                "frames. RBuffer contains {2}/{3} active episodes, "
                "{4} total frames and {5} registered frames.".format(
                    self._current_episode.episode_num,
                    self._current_episode.length,
                    self.total_active_episodes, self.total_episodes,
                    self.total_active_trans, self.__len__()))
            self._current_episode = None

    def _debug(self, steps, output_data):
        self.writer.add_scalar(
            tag="{0}_replay/total_frames".format(self._tag),
            scalar_value=output_data["frames"]["totals"],
            global_step=steps)
        self.writer.add_scalar(
            tag="{0}_replay/total_active_frames".format(self._tag),
            scalar_value=output_data["frames"]["total_actives"],
            global_step=steps)
        self.writer.add_scalar(
            tag="{0}_replay/total_active_episodes".format(self._tag),
            scalar_value=output_data["episodes"]["total_actives"],
            global_step=steps)
        if self._current_episode is None:
            last_episode_num = self._buffer[-1].episode_num
        else:
            last_episode_num = self._buffer[-2].episode_num
        _active_epi = [self._convert_episode_idx(i) for i in range(
            self._buffer[self._first_active_idx].episode_num, last_episode_num)]
        self.writer.add_histogram(
            tag="{0}_replay/episodes_range".format(self._tag),
            values=np.arange(self._buffer[self._first_active_idx].episode_num,
                             last_episode_num+1),
            global_step=steps)
        _epi_length = np.array([i.length for i in _active_epi])
        _epi_rewards = np.array([i.episode_total_reward for i in _active_epi])
        self.writer.add_histogram(
            tag="{0}_replay/epi_length".format(self._tag),
            values=_epi_length,
            global_step=steps)
        self.writer.add_histogram(
            tag="{0}_replay/epi_reward".format(self._tag),
            values=_epi_rewards,
            global_step=steps)

    def debug(self, steps, n_samples_tsne=1000):
        """Understand how the replay buffer is composed.

        NOTE: ignore `episodes[start_num]` and `episodes[end_num]`. I originally
        used these to track the beginning and ending episodes for the loaded
        samples from the teacher replay buffers, but it's not going to be
        accurate as it takes the first and last episodes in the buffer, whereas
        I add then in a 'snake' fashion where we branch out from a central
        episode.
        """
        output_data = {
            "episodes": {
                "start_num":
                    self._buffer[self._first_active_idx].episode_num,
                "end_num": self._buffer[-1].episode_num,
                "totals": self.total_episodes,
                "total_actives": self.total_active_episodes,
            },
            "frames": {
                "totals": self.total_active_trans,
                "total_actives": self.__len__(),
            },
            "current_pos": self._current_pos,
            "last_episode": {
                "incomplete": self._current_episode is not None,
                "length": self._buffer[-1].length,
                "rewards": self._buffer[-1].episode_total_reward,
            },
        }
        # self._debug(steps, output_data)
        if "p" in self.info:
            output_data["progress_scores"] = self.debug_progress(
                steps, n_samples_tsne)
        if self._debug_dir:
            self.logger.debug(pformat(output_data))
            self._debug_output[steps] = output_data
            write_dict(self._debug_output, self._debug_dir, "summary")
        else:
            return output_data

    def _debug_progress(self, _output, steps, n_samples_tsne):
        self.writer.add_histogram(
            tag="{0}/progress_scores".format(self._tag),
            values=self.info["progress_scores"],
            global_step=steps)
        self.writer.add_histogram(
            tag="{0}/progress_scores_abs".format(self._tag),
            values=self.info["_progress_scores"],
            global_step=steps)
        self.writer.add_scalar(
            tag="{0}/p_perplexity".format(self._tag),
            scalar_value=_output["perplexity"],
            global_step=steps)
        # sample with the extreme values
        progrss_scores_idx = self.info["_progress_scores"].argsort()
        _mid_point = int(len(progrss_scores_idx) / 2)
        _left_mid_point = _mid_point - int(n_samples_tsne / 2)
        _right_mid_point = _mid_point + int(n_samples_tsne / 2)
        tsne_sample_idx = progrss_scores_idx[
            np.r_[0:n_samples_tsne, _left_mid_point:_right_mid_point,
                  (-n_samples_tsne):0]]
        _transitions = generate_top_transitions_grid(
            n=4, sorted_index=progrss_scores_idx, replay=self,
            num_steps=4, top=True)
        self.writer.add_image(
            tag="{0}/top_progress_scores".format(self._tag),
            img_tensor=vutils.make_grid(torch.from_numpy(_transitions),
                                        normalize=True, scale_each=True),
            global_step=steps)
        _transitions = generate_top_transitions_grid(
            n=4, sorted_index=progrss_scores_idx, replay=self,
            num_steps=4, top=False)
        self.writer.add_image(
            tag="{0}/bottom_progress_scores".format(self._tag),
            img_tensor=vutils.make_grid(torch.from_numpy(_transitions),
                                        normalize=True, scale_each=True),
            global_step=steps)
        _tsne_plot = np.transpose(
            plot_tsne(sample_index=tsne_sample_idx, replay=self),
            (2, 0, 1)) / 260.0
        self.writer.add_image(
            tag="{0}/tsne".format(self._tag),
            img_tensor=torch.from_numpy(_tsne_plot),
            global_step=steps)

    def debug_progress(self, steps, n_samples_tsne=1000):
        _output = {
            "perplexity": relative_perplexity(self.info["p"]),
            "mean_progress_scores": self.info["progress_scores"].mean(),
        }
        # self._debug_progress(_output, steps, n_samples_tsne)
        return _output


class SharedExperienceReplay(ExperienceReplay):
    def __init__(self, writer, capacity, init_cap, frame_stack, gamma, tag,
                 debug_dir=None):
        super().__init__(writer=writer, capacity=capacity, init_cap=init_cap,
                         frame_stack=frame_stack, gamma=gamma, tag=tag,
                         debug_dir=debug_dir)

    def add_teacher_episode(self, episode, teacher_id):
        assert isinstance(episode, Episode)
        self._register_one_teacher_episode(
            next_episode_idx=self.total_episodes,
            episode=episode, teacher_id=teacher_id)
        self._add_one_episode(episode)

    def _register_one_teacher_episode(self, next_episode_idx, episode,
                                      teacher_id):
        assert isinstance(episode, Episode)
        self._episode_lookup[(teacher_id, episode.episode_num)] = \
            next_episode_idx
        for i in range(episode.length):
            self._frame_lookup[self._current_pos] = (
                teacher_id, episode.episode_num, i)
            self._current_pos = self._current_pos + 1

    def _convert_teacher_episode_idx(self, teacher_id, episode_num):
        assert (teacher_id, episode_num) in self._episode_lookup
        return self._buffer[self._episode_lookup[(teacher_id, episode_num)]]

    def _get_transition(self, i, num_steps, force=False):
        assert 0 <= i < self.__len__()
        _teacher_id, _episode_num, _frame_num = self._frame_lookup[i]
        return self._convert_teacher_episode_idx(_teacher_id, _episode_num)\
            .sample_transition(
                idx=_frame_num, frame_stack=self._frame_stack,
                num_steps=num_steps, gamma=self._gamma, force=force)

    def add_replay(self, replay, teacher_id):
        for episode in replay.buffer:
            self.add_teacher_episode(episode, teacher_id)
        for keys, items in replay.info.items():
            if keys not in self.info:
                self.info[keys] = items
            else:
                self.info[keys] = np.concatenate([self.info[keys], items])


class ExperienceSource:

    def __init__(self, env, agent, episode_per_epi):
        """
        `ExperienceSource` is an iterable that integrates environment and agent.
        In train/test processes, we call the `next` for stepping purposes.

        :param env: an `Environment` object, custom class wrapping around gym
            env, adds stuff for saving; see `dqn/environment/setup.py`. Exposes
            a similar `step` interface which calls the true gym env's step.
        :param agent: an `Agent` object
        :param episode_per_epi: Number of episodes to write to disk.
        """
        assert isinstance(agent, Agent)
        assert isinstance(env, Environment)
        self.env = env
        self.agent = agent
        self.episode_per_epi = episode_per_epi
        self.mean_reward = None
        self.latest_speed = None
        self.latest_reward = None

    def __iter__(self):
        """THIS is what calls `finish_episode` with `save_tag`.
        """
        while True:
            _obs = self.env.env_obs
            _steps = self.env.env_steps
            action = self.agent(_obs, self.env.total_steps)
            self.env.step(action)  # ENVIRONMENT STEPPING!!!
            _next_obs = np.expand_dims(self.env.env_obs[-1], 0)
            if _steps == 0:
                # new episodes, need `state`
                _obs = _obs
            else:
                # --------------------------------------------------------------
                # Existing episodes, set `state` to None.  XP replay code
                # w/`Transition`s don't check `state` except for when it's the
                # first in episode: see `add_one_transition()` above.
                # --------------------------------------------------------------
                _obs = None
            transition = Transition(state=_obs, next_state=_next_obs,
                                    action=action, reward=self.env.env_rew,
                                    done=self.env.env_done)
            yield transition
            # ------------------------------------------------------------------
            # If env (either train or test) has an episode which just finished,
            # call this to formally finish and potentially save the trajectory.
            # The `episode_per_epi` determines save frequency. Can increase it
            # to decrease memory requirements. But this should only determine
            # saving trajs into pickle files; regardless of what happens the
            # replay _buffer_ should get the frames.
            # ------------------------------------------------------------------
            # Recall yield keyword: we stop at transition above, then next time
            # this is called, we start here to check for if we lost a life.
            # ------------------------------------------------------------------
            if self.env.env_done:
                if self.episode_per_epi:
                    save_tag = self.env.get_num_lives() % self.episode_per_epi == 0
                else:
                    save_tag = False
                self.latest_reward = self.env.epi_rew
                self.env.finish_episode(
                    save=save_tag, gif=False,
                    epsilon=self.agent.get_policy_epsilon(self.env.total_steps))
                self.latest_speed = self.env.speed
                self.mean_reward = self.env.mean_rew

    def pop_latest(self):
        r = self.latest_reward
        mr = self.mean_reward
        s = self.latest_speed
        if r is not None:
            self.latest_reward = None
            self.mean_reward = None
            self.latest_speed = None
        return r, mr, s
