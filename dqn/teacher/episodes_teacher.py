from dqn.replay.transition import Transition
from dqn.teacher.teacher import Teacher
from dqn.utils.summary import IterativeSummary, MatchingSummary
from dqn.utils import io
import numpy as np


class EpisodesIterativeTeacher(Teacher):
    """We don't use, see `EpisodesMatchingTeacher` below."""
    def __init__(self, writer, model_name, teacher_num, train_params,
                 gpu_params, teacher_params, env_params, log_params, debug_dir):
        super().__init__(
            writer=writer, model_name=model_name, teacher_num=teacher_num,
            train_params=train_params, gpu_params=gpu_params,
            teacher_params=teacher_params, env_params=env_params,
            log_params=log_params, debug_dir=debug_dir)
        self.logger.info("Teacher {0} has been set to be an iterative episode "
                         "teacher.".format(self.teacher_num))
        # `_latest_episode` serves as a temporary buffer to hold the episode
        # memory. Once it all loads to the replay buffer, it needs to be set
        # to `None` in order to read the next episode in.
        self._latest_episode = None
        # self._next_episode_number = self.get_init_next_episode_number()
        self._latest_episode_next_offset = 0
        # Maintain a replay buffer

    def init_teacher(self):
        self._episodes = IterativeSummary(
            model_dir=self._model_dir, summary_type="episodes",
            target_field="total_rew",
            lead_number=self._teacher_params["num_episode_ahead"],
            training=True)
        self._create_replay()
        self._replay._buffer = [None] * self._teacher_params[
            "num_episode_ahead"]
        self._replay._first_active_idx = self._teacher_params[
            "num_episode_ahead"]

    @property
    def done_reading_all_episodes(self):
        return self._episodes.done

    @property
    def done_reading_current_episode(self):
        """
        :return: if teacher has finished reading current episode. There are
        two signs that the currently episode is done,
            (1) `_latest_episode` is None
            (2) `_latest_episode_next_offset` reaches to the maximum for the
            current episode
        """
        if self._latest_episode is None:
            return True
        return self._latest_episode_next_offset >= self._latest_episode.length

    def _read_next_episode(self):
        """
        Read the next episode into teacher's replay memory
        """
        if self.done_reading_all_episodes:
            return
        assert self.done_reading_current_episode
        _next_episode_num = self._episodes.next()
        self._latest_episode = self._read_episode(_next_episode_num)
        self._latest_episode_next_offset = 0

    def _read_and_load_one_transition(self):
        """
        Read one sample into teacher's replay memory. This process is
        superficial as the memory is already loaded. Essentially, we only
        change the offset in the replay buffer for sampling purposes
        """
        if self.done_reading_all_episodes:
            return
        if self.done_reading_current_episode:
            self._read_next_episode()
        _frame_stack = self._env_params["frame_stack"]
        # For the first transition in the episode, it is special
        if self._latest_episode_next_offset == 0:
            _state = self._latest_episode.states[0:_frame_stack]
        else:
            _state = None
        _next_state = np.expand_dims(self._latest_episode.states[
            self._latest_episode_next_offset + _frame_stack], axis=0)
        _action = self._latest_episode.actions[self._latest_episode_next_offset]
        _reward = self._latest_episode.rewards[self._latest_episode_next_offset]
        self._latest_episode_next_offset += 1
        _done = self._latest_episode_next_offset == self._latest_episode.length
        self._replay.add_one_transition(Transition(
            state=_state, next_state=_next_state, action=_action,
            reward=_reward, done=_done))

    def read_n_transitions(self, n):
        for i in range(n):
            self._read_and_load_one_transition()

    def get_learner_weights(self, obs, acts):
        return None

    def get_teacher_samples(self, batch_size, num_steps, steps):
        transitions = self._replay.sample(batch_size, num_steps=num_steps)
        return transitions


class EpisodesMatchingTeacher(Teacher):

    def __init__(self, writer, model_name, teacher_num, train_params,
                 gpu_params, teacher_params, env_params, log_params,
                 debug_dir, opt_params):
        """Parent of SnapshotsTeacher, use for `load_episodes_in_window()`.
        """
        self._opt_params = opt_params
        super().__init__(
            writer=writer, model_name=model_name, teacher_num=teacher_num,
            train_params=train_params, gpu_params=gpu_params,
            teacher_params=teacher_params, env_params=env_params,
            log_params=log_params, debug_dir=debug_dir)
        self.logger.info("Teacher {0} has been set to be a matching episode "
                         "teacher.".format(self.teacher_num))

    def init_teacher(self):
        self._episodes = MatchingSummary(
            model_dir=self._model_dir, summary_type="episodes",
            target_field="total_rew",
            lead_number=self._teacher_params["num_episode_ahead"],
            training=True,
            smooth_window=self._teacher_params["rew_avg_window"])

    @property
    def done_reading(self):
        return self._episodes.done

    def get_learner_weights(self, obs, acts):
        return None

    def get_teacher_samples(self, batch_size, num_steps, steps):
        if self._replay is None:
            return None
        transitions = self._replay.sample(batch_size, num_steps=num_steps)
        return transitions

    def load_episodes_in_window(self, matched_episode: int):
        """Loads lifespans in window around specified episode_num (not lives).

        SnapshotsTeacher.load_transitions() calls this, to load states.

        Earlier, I thought this: pickle file numbers are 1-indexed, while
        episodes are 0-indexed.  Additionally, games have varying numbers of
        lives. An "episode" corresponds to all the lives for a single game, so
        if a game has k lives, then there are k corresponding pickle files.
        Therefore if `episode_idx` is the index of the episode to load, we want
        to load:

            for i in range(k):
                filename = episode_train_{padded(episode_idx * k + 1)}.pkl

        Alas! It is not true. https://github.com/CannyLab/dqn/issues/44
        This means we have to refer to the teacher's original training summary
        file, and then infer the lifespan that started and ended the episode.

        Parameters
        ---------
        matched_episode: int
            The episode number of a matched snapshot. Episodes will be loaded
            in a window surrounding this episode. This should be called from
            the number of finished episodes from the snapshots df. Do NOT get
            this from summing over the episode steps in the episode df! Also,
            this is the actual true episode, not the number of lives.
        """
        # Create new xp replay, deleting older one if it exists.
        self._create_replay()

        lives = self.total_lives if self.total_lives != 0 else 1
        max_samples = self._teacher_params['num_teacher_samples']
        lg = self.logger

        def load_lives_from_episode(episode_idx: int):
            life_beg = self._episodes['life_idx_begin', episode_idx]
            life_end = self._episodes['life_idx_end', episode_idx]
            # Don't forget the +1, and ALSO, lives above are 1-indexed, so
            # (annoyingly), we subtract one for input to `number_0_idx` args.
            for life in range(life_beg, life_end + 1):
                lifespan = io.read_episode(self._model_dir, number_0_idx = life - 1)
                self._replay.add_episode(lifespan)

        index_before_match = matched_episode - 1
        index_after_match = matched_episode

        # Alternate between loading episodes before and after matched episode.
        while len(self._replay) < max_samples:
            if index_before_match >= 0:
                load_lives_from_episode(index_before_match)
                index_before_match -= 1
            if index_after_match < len(self._episodes):
                load_lives_from_episode(index_after_match)
                index_after_match += 1
            if index_before_match < 0 and index_after_match >= len(self._episodes):
                lg.warn("Breaking, no more valid indices to load episodes. Should never happen!")
                break
        lg.info("Length of replay w/teacher samples: {}".format(len(self._replay)))

    def matching(self, latest_reward, steps, agent=None):
        """Will throw an error since episodes df doesn't have `cum_steps` now."""
        if self.done_reading:
            return
        _next_episode_num = self._episodes.next(latest_reward)
        if _next_episode_num is not None:
            lm = self._episodes.last_match_idx
            lr = self._episodes.last_read_idx
            self.logger.info(
                "Agent ({0:.2f} rew, {1} cum steps) has matched to episode {2} "
                "({3:.2f} rew, {4} cum steps). By ZPD, it targets episode {5} "
                "({6:.2f} rew, {7} cum steps). ".format(
                    latest_reward, steps,
                    self._episodes["number", lm],
                    self._episodes[self._episodes.target_field, lm],
                    self._episodes["cum_steps", lm],
                    self._episodes["number", lr],
                    self._episodes[self._episodes.target_field, lr],
                    self._episodes["cum_steps", lr]))
            self.load_episodes_in_window(episode_num=int(_next_episode_num))
