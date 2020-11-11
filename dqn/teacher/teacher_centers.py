from dqn.utils.schedules import PiecewiseSchedule, ConstantSchedule, \
    LinearSchedule
from dqn.teacher.episodes_teacher import EpisodesMatchingTeacher, \
    EpisodesIterativeTeacher
from dqn.teacher.weight_teacher import WeightTeacher
from dqn.teacher.snapshots_teacher import SnapshotsTeacher
from dqn.utils.variables import create_var
from dqn.utils.experience import merge_transitions_xp
from dqn.replay.experience import SharedExperienceReplay
from dqn.utils.math import softmax_deepmind as boltzmann
import torch
import numpy as np
import logging
import os
import time

# Use this in `_identify_teaching_method` to create teachers.
teacher_mapping = {
    "weight": WeightTeacher,
    "episodes-matching": EpisodesMatchingTeacher,
    "episodes-iterative": EpisodesIterativeTeacher,
    "snapshots": SnapshotsTeacher
}


class TeachingCenter:
    def __init__(self):
        self.teacher_params = None
        self.teacher_weights = None
        self._env_params = None
        self._log_params = None

    def timer(self):
        pass

    def reset_timer(self):
        pass

    @property
    def total_num_teachers(self):
        raise NotImplementedError

    @property
    def supervise_loss(self):
        if self.teacher_params is None:
            return False
        if self.teacher_params["type"] != "weight":
            return self.teacher_params["supervise_loss"]["enabled"]
        else:
            return False

    @property
    def supervise_loss_lambda(self):
        assert self.supervise_loss
        return self.teacher_params["supervise_loss"]["lambda"]

    @property
    def supervise_margin(self):
        assert self.supervise_loss
        return self.teacher_params["supervise_loss"]["margin"]

    @property
    def supervise_type(self):
        assert self.supervise_loss
        return self.teacher_params["supervise_loss"]["type"]

    @property
    def is_weight_teacher(self):
        if self.teacher_params is None:
            return False
        return self.teacher_params["type"] == "weight"

    def read_n_transitions(self, n):
        raise NotImplementedError

    def matching(self, latest_reward, steps, agent):
        raise NotImplementedError

    def weight(self, obs, actions):
        raise NotImplementedError

    def sample(self, batch_size, steps, num_steps=None):
        raise NotImplementedError

    def sample_teacher(self, active_teachers):
        assert 1 <= len(active_teachers) <= self.total_num_teachers
        _subsample_p = self.teacher_weights[active_teachers]
        _subsample_p = _subsample_p / _subsample_p.sum()
        return np.random.choice(active_teachers, 1, p=_subsample_p)[0]

    @property
    def eta(self):
        t = self._env_params["max_num_steps"] / \
            self._log_params["debug_per_step"]
        return min(0.5, np.sqrt(np.log(self.total_num_teachers) / t))


class MultiTeacherTeachingCenter(TeachingCenter):

    def __init__(self, writer, teacher_params, train_params, gpu_params,
                 env_params, log_params, debug_dir, opt_params):
        """Teacher center that controls an ensemble of teachers.

        Used in `atari.py` for teachers, even if we've got just one teacher, or
        even if there's _no_ teacher.

        - self.active: for if we use teachers at all. It's False if we have no
          teachers or if the agent has surpassed the best teacher snapshot.
          (Since it applies with and w/out teachers, I use `total_num_teachers`
          below for detecting if I am using any teachers)
        - self._teachers: dict with one element per teacher
        """
        super().__init__()
        self.teacher_params = teacher_params
        self._train_params = train_params
        self._gpu_params = gpu_params
        self._env_params = env_params
        self._log_params = log_params
        self._opt_params = opt_params
        self._teachers = {}
        self.logger = logging.getLogger("teaching_control")
        self.active = True
        self.writer = writer
        self._debug_dir = debug_dir
        if debug_dir and not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        # `EpisodesTeacher` always weighs learner's sample equally
        self._train_batch_ones = create_var(
            train_params["batch_size"], gpu=self._gpu_params["enabled"],
            gpu_id=self._gpu_params["id"], convert_func="torch_ones")
        self._identify_teaching_method()
        self.teacher_weights = np.ones(self.total_num_teachers)

    @property
    def total_num_teachers(self):
        """Use this in external code to detect if we aren't using teachers."""
        return len(self._teachers)

    def any_teacher_done(self):
        """For detecting if we've hit last teacher.

        TODO: need to get this working with multiple teachers. Actually since
        even in the multi-teacher case (will have to check) we pick one teacher,
        THEN do the sampling (see above), we can merge this with `sample`. Think
        carefully, right now we use self.active ... but the definition of that
        should be, when we stop matching (NOT as it was defined earlier, when
        we've stopped finding new ZPD stuff ... because at the end we are still
        matching even if the ZPD doesn't update itself, because we want progress
        scores to continually get updated).
        """
        return not self.active

    def _identify_teaching_method(self):
        """Identify teaching method: "weighting", "episodes", "snapshots".

        Assigns to `self._teachers`, a dict with one element per teacher. The
        `teacher_params['models']` is a list of teacher snapshots. Also done for
        `SnapshotDistillation` subclass as it calls the super `init()`.  The
        `teacher_mapping` will _form_ the repsective teacher classes (e.g.,
        `SnapshotsTeacher`) to be stoerd into `self._teachers`.
        """
        if self.teacher_params is None:
            self.active = False
        else:
            assert self.teacher_params["type"] in teacher_mapping
            model_names = self.teacher_params["models"]
            self.logger.info(
                "{0} learning algorithm has been detected with teachers "
                "{1}".format(self.teacher_params["type"], model_names))
            for _name in model_names:
                _init_params = {
                    "writer": self.writer,
                    "model_name": _name,
                    "teacher_num": self.total_num_teachers,
                    "train_params": self._train_params,
                    "gpu_params": self._gpu_params,
                    "teacher_params": self.teacher_params,
                    "env_params": self._env_params,
                    "log_params": self._log_params,
                    "debug_dir": os.path.join(
                        self._debug_dir,
                        "teacher_{0}".format(self.total_num_teachers))
                }
                if self.teacher_params["type"] == "episodes-matching" or \
                        self.teacher_params["type"] == "snapshots":
                    _init_params["opt_params"] = self._opt_params
                _teacher = teacher_mapping[self.teacher_params["type"]](
                    **_init_params)
                self._teachers[self.total_num_teachers] = _teacher

            if self.teacher_params["type"] != "weight":
                # The fraction of the TEACHER SAMPLES in each minibatch.
                start = self.teacher_params["blend"]["start"]
                end   = self.teacher_params["blend"]["start"]
                self.blend_schedule = PiecewiseSchedule([
                    (0,                                      start),
                    (self.teacher_params["blend"]["frames"],   end)
                ], outside_value=end)

                # Heuristic, because I was getting performance drops if we make
                # true_end=0 for Breakout, Seaquest, and SInvaders.  Hopefully
                # this is not a problem for other envs/agents.
                true_end = 0.05

                # `start` and `end` are fraction of the minibatch of TEACHER
                # samples; student gets the rest. For 25%, we get [8,24] split
                # for [t,s]. For 50% it's [16,16], for 75% it's [24,8].
                assert start == end, 'For now use a fixed ratio, except for decay period'

                # Careful, this decay rate is called as many times as there are
                # gradient updates, which change according to the ratio.  If we
                # had 100% student (or self-generated data), the tf would be 4.
                _tf = self._train_params['train_freq_per_step']

                # For the number I use to arrive at `true_end`, think about how
                # many times we call this. For 50% teacher, we call this 500k
                # times (500k mbs), and 2 step : 1 mb means 1000K=1M env steps.
                if start == 0.25:
                    assert _tf == 3, '{} vs {}'.format(start, _tf)
                    self.blend_schedule_end = PiecewiseSchedule([
                        (0,      end),
                        (333333, true_end)
                    ], outside_value=true_end
                    )
                elif start == 0.50:
                    assert _tf == 2, '{} vs {}'.format(start, _tf)
                    self.blend_schedule_end = PiecewiseSchedule([
                        (0,      end),
                        (500000, true_end)
                    ], outside_value=true_end
                    )
                elif start == 0.75:
                    assert _tf == 1, '{} vs {}'.format(start, _tf)
                    self.blend_schedule_end = PiecewiseSchedule([
                        (0,       end),
                        (1000000, true_end)
                    ], outside_value=true_end
                    )
                else:
                    raise ValueError(start)

    def read_n_transitions(self, n):
        if not self.active or self.teacher_params["type"] != \
                "episodes-iterative":
            return
        _any_active = False
        for i in range(self.total_num_teachers):
            teacher = self._teachers[i]
            if not teacher.done_reading_all_episodes:
                teacher.read_n_transitions(n)
                _any_active = True
        if not _any_active:
            self.logger.info("Episode learning stopped as teacher doesn't "
                             "have episode history any more")
            self.active = False

    def matching(self, latest_reward, steps, agent=None):
        """Iterates through all teachers and (for each) find matching snapshots.

        Called from `atari.py` after each episode (from student) concludes.
        The teacher_params['type'] is usually 'snapshots', and `active` guards
        in case we don't use/need a teacher.

        Parameters
        ----------
        latest_reward: the past average of TRUE (not clipped) rewards
        steps: number of training steps
        agent: Optional, pass the agent to refernece it in case.

        Returns
        -------
        Either None, or a list of matched indices, to make it easier from
        `atari.py` to tell if we did some matching. Though, often the matched
        list will itself be [None]. Note: this is purely for debugging.
        """
        if not self.active or self.teacher_params["type"] == "weight" \
                 or self.teacher_params["type"] == "episodes-iteratve":
            return None
        _any_active = False
        matched = []
        for i in range(self.total_num_teachers):
            teacher = self._teachers[i]
            if not teacher.done_reading:
                zpd_idx = teacher.matching(latest_reward, steps, agent=agent)
                matched.append(zpd_idx)
                _any_active = True
        if not _any_active:
            self.logger.info("Snapshot learning stopped at steps {0} as "
                             "teacher doesn't have snapshot history any "
                             "more".format(steps))
            self.active = False
        return matched

    def weight(self, obs, actions):
        """
        Given the current states and actions, weight it according to certain
        heuristics. It uses the default setup in the settings. If not set up
        before, it returns a vector of ones.

        :param obs: A batch of input observations
        :param actions: A batch of input actions
        :return: The weights for the batch averaged across all teachers
        """
        if self.teacher_params and self.teacher_params["type"] == "weight":
            _batch_default_type = obs.volatile
            obs.volatile = True
            scores = []
            for teacher in self._teachers:
                score = teacher.get_learner_weights(obs, actions)
                # standard it to center at 1
                score = (score - score.mean()) / score.std() + 1.0
                # no negative weighting
                score[score <= 0] = 0.0
                score.volatile = False
                scores.append(score)
            obs.volatile = _batch_default_type
            return torch.stack(scores).mean(dim=0)
        else:
            return self._train_batch_ones

    def sample(self, batch_size, steps, num_steps=None):
        """
        Sample `batch_size` experience from teacher with `teacher_id`. If
        learner_samples are provided, blend teacher samples along with them.

        Looks like we pick the teacher to sample, then we pick samples from that
        teacher only.

        :param batch_size: total number of samples in the batch to sample
        :param steps: current frame index
        :param num_steps: a list of num_steps samples to retrieve
        :return: A tuple of samples for training
        """
        if batch_size > 0:
            teacher_id = self.sample_teacher(range(self.total_num_teachers))
            return self._teachers[teacher_id].get_teacher_samples(
                batch_size, num_steps, steps), teacher_id
        else:
            return None, None


class MultiAgentTeachingCenter(TeachingCenter):
    def __init__(self, writer, exp_queues, end_signals, learner_id, env_params,
                 teacher_params, train_params, gpu_params, log_params):
        super().__init__()
        self.teacher_params = teacher_params
        self._env_params = env_params
        self._log_params = log_params
        self._exp_queues = exp_queues
        self._end_signals = end_signals
        self._learner_id = learner_id
        self._train_params = train_params
        self._gpu_params = gpu_params
        self.blend_schedule = PiecewiseSchedule([
            (0, self.teacher_params["blend"]["start"]),
            (self.teacher_params["blend"]["frames"],
             self.teacher_params["blend"]["end"])
        ], outside_value=self.teacher_params["blend"]["end"])
        self.logger = logging.getLogger("teaching_control_{0}".format(
            learner_id))
        self.active = True
        self.writer = writer
        # Always weighs learner's sample equally
        self._train_batch_ones = create_var(
            train_params["batch_size"], gpu=self._gpu_params["enabled"],
            gpu_id=self._gpu_params["id"], convert_func="torch_ones")
        self.teacher_weights = np.ones(self.total_num_teachers)
        self._download = np.zeros(len(exp_queues))
        self._download_time = np.zeros(len(exp_queues))

    def timer(self):
        return {
            "download_amount": list(self._download),
            "download_time": list(self._download_time)
        }

    def reset_timer(self):
        self._download = np.zeros(len(self._exp_queues))
        self._download_time = np.zeros(len(self._exp_queues))

    @property
    def total_num_teachers(self):
        return len(self._exp_queues)

    @property
    def is_weight_teacher(self):
        return False

    def read_n_transitions(self, n):
        pass

    def matching(self, latest_reward, steps, agent=None):
        pass

    def weight(self, obs, actions):
        return self._train_batch_ones

    def sample(self, batch_size, steps, num_steps=None):
        if batch_size > 0:
            _active_teachers = []
            for i in range(self.total_num_teachers):
                if i == self._learner_id or self._end_signals[i].is_set() or \
                        self._exp_queues[i].empty():
                    pass
                else:
                    _active_teachers.append(i)
            if len(_active_teachers) < self.total_num_teachers - 1:
                self.logger.info("Only {0} teachers are active now".format(
                    _active_teachers))
            if len(_active_teachers) == 0:
                return None, None
            teacher_id = self.sample_teacher(_active_teachers)
            if self._exp_queues[teacher_id].qsize() <= batch_size:
                return None, None
            _transitions = []
            _start_time = time.time()
            for _ in range(batch_size):
                _transition = self._exp_queues[teacher_id].get()
                _transitions.append(_transition)
            self._download_time[teacher_id] += time.time() - _start_time
            self._download[teacher_id] += batch_size
            _transitions = merge_transitions_xp(_transitions)
            return _transitions, teacher_id
        else:
            return None, None


class SnapshotDistillation(MultiTeacherTeachingCenter):
    """Called with `atari-multi-teacher.py`, not with `atari.py`.
    """
    def __init__(self, writer, teacher_params, train_params, gpu_params,
                 env_params, log_params, debug_dir, opt_params):
        super().__init__(writer=writer, teacher_params=teacher_params,
                         train_params=train_params, gpu_params=gpu_params,
                         env_params=env_params, log_params=log_params,
                         debug_dir=debug_dir, opt_params=opt_params)
        assert self.teacher_params["type"] == "snapshots"
        self._replay = None
        self.beta_schedule = LinearSchedule(
            schedule_timesteps=self._env_params["max_num_steps"] //
            self._train_params["train_freq_per_step"],
            final_p=1.0,
            initial_p=self.teacher_params["init_wis_beta"])

    def _match_teacher(self, latest_reward, steps):
        """D: called in each `self._matching` call.

        Returns LIST of matched snapshot idx from _each_ teacher. Uses `None` if
        we've gone beyond the snapshots, so that the list length is consistent.
        But, if not `None`, then we load transitions, thus creating a new replay
        buffer (`ExperienceReplay` class) for each teacher, internally.

        Differs from the one in multi-teacher center, since that one doesn't
        use the teacher.snapshots? So, e.g., it doesn't track indices? But that
        functionality cam come later in `teacher.matching`, whereas we don't use
        that here?
        """
        _matching_snapshot_num = []
        for teacher_id, teacher in self._teachers.items():
            # Record teacher's last match snapshot number
            _last_snapshot_num = teacher.snapshots.condensed_get_field(
                teacher.snapshots.last_match_idx, "number")
            # Match learner's current reward with each teacher
            _current_match_num = teacher.snapshots.match_with_rewards(
                latest_reward)
            _matching_snapshot_num.append(_current_match_num)
            if _current_match_num is None:
                # reward has gone beyond teacher's history
                self.logger.info(
                    "Teacher {0} doesn't have snapshots that contain rewards "
                    "higher than {1}".format(teacher_id, latest_reward))
                continue
            # Contribute samples, which are loaded from
            # `_snapshot.last_read_idx`
            _current_reach_num = teacher.snapshots.condensed_get_field(
                teacher.snapshots.last_read_idx, "number")
            teacher.load_transitions(latest_reward, steps, _current_reach_num,
                                     _last_snapshot_num)
        return _matching_snapshot_num

    def _score_teacher_replays(self, steps, matching_snapshot_num, dedup=False):
        """D: called in each `self._matching` call, after `self._match_teacher`.

        Returns
        -------
        _num_replay_to_use:
        """
        _num_replay_to_use = 0
        if dedup:
            _batch = self.sample(
                batch_size=self._train_params["batch_size"],
                steps=None, num_steps=self._train_params["num_steps"])[0]
            for teacher_id, teacher in self._teachers.items():
                if matching_snapshot_num[teacher_id] is not None:
                    teacher.progress_step(steps=steps, transitions=_batch)

        for teacher_id, teacher in self._teachers.items():
            _progress_scores = np.zeros(len(teacher.replay))
            _activations = np.zeros((len(teacher.replay),
                                     self._train_params["hidden_size"]))
            _num_scorer = 0
            for j in range(len(self._teachers)):
                if teacher_id == j:
                    continue
                _current_match_num = matching_snapshot_num[j]
                if _current_match_num is None:
                    # not score any samples
                    pass
                else:
                    if dedup:
                        _scores = self._teachers[j].generate_progress_score(
                            transitions=teacher.replay)
                    else:
                        _scores = self._teachers[j].progress_score(
                            _current_match_num, transitions=teacher.replay)
                    _num_scorer += 1
                    _progress_scores += _scores["progress_scores"]
                    _activations += _scores["activations"]
            if _num_scorer >= 2:
                teacher.replay.info = {
                    "progress_scores": _progress_scores / _num_scorer,
                    "activations": _activations / _num_scorer
                }
                _num_replay_to_use += 1
            else:
                teacher.replay.info = None
        return _num_replay_to_use

    def _combine_replay(self, steps):
        if self._replay is not None:
            del self._replay
        self._replay = SharedExperienceReplay(
            writer=self.writer,
            capacity=self.teacher_params["replay_size"],
            init_cap=0,
            frame_stack=self._env_params["frame_stack"],
            gamma=self._train_params["gamma"],
            tag="teacher",
            debug_dir=None)

        for i in range(len(self._teachers)):
            _teacher_replay = self._teachers[i].replay
            if _teacher_replay.info is None:
                continue
            _batch_scores = _teacher_replay.info["progress_scores"].copy()
            _teacher_replay.info["progress_signs"] = np.sign(_batch_scores)
            if not self.teacher_params["negative_correction"]:
                _batch_scores[_batch_scores <= 0] = self.teacher_params[
                    "progress_epsilon"]
            else:
                _batch_scores = np.abs(_batch_scores)
            _teacher_replay.info["_progress_scores"] = _batch_scores
            self._replay.add_replay(_teacher_replay, i)
        self._replay.info["p"] = boltzmann(
            self._replay.info["_progress_scores"],
            T=self.teacher_params["temperature"])
        self.logger.info(self._replay.debug_progress(steps=steps))

    def matching(self, latest_reward, steps, agent=None):
        if not self.active:
            return
        _matching_snapshot_num = self._match_teacher(latest_reward, steps)
        _num_replay_to_use = self._score_teacher_replays(
            steps, _matching_snapshot_num, dedup=False)
        if _num_replay_to_use >= 1:
            self._combine_replay(steps)
            for i in range(self.teacher_params["dedup_iterations"]):
                self._score_teacher_replays(
                    steps=steps,
                    matching_snapshot_num=_matching_snapshot_num,
                    dedup=True)
                self._combine_replay(steps)
        else:
            self.active = False

    def sample(self, batch_size, steps, num_steps=None):
        if self._replay is None:
            return None, None
        if batch_size > 0:
            transitions, idx = self._replay.sample(
                batch_size, num_steps=num_steps, output_idx=True)
            _weights = transitions.weight
            if self.teacher_params["negative_correction"]:
                _weights *= self._replay.info["progress_signs"][idx]
            if "p" in self._replay.info:
                _wis = np.power(self._replay.info["p"][idx] * len(self._replay),
                                -self.beta_schedule.value(steps))
                _wis /= _wis.max()
                _weights *= _wis
            transitions.weight = _weights
            return transitions, None
        else:
            return None, None
