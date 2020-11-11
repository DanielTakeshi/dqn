from dqn.utils.io import read_json, read_snapshot, read_episode
from dqn.utils.train import init_atari_model
from dqn.replay.experience import ExperienceReplay
import os
import logging
from os.path import join
from dqn import common_config as cfg


class Teacher:

    def __init__(self, writer, model_name, teacher_num, train_params,
                 gpu_params, teacher_params, env_params, log_params, debug_dir):
        """Make teachers (`init_teacher`), which loads summaries as panda dfs.

        For identifying *student* directories, you want the `log_params`.
        """
        self._model_dir = join(cfg.SNAPS_TEACHER, model_name)
        self.teacher_num = teacher_num
        self.writer = writer
        self._train_params = train_params
        self._gpu_params = gpu_params
        self._env_params = env_params
        self._log_params = log_params
        self._teacher_params = teacher_params
        self._params = dict(read_json(
            dir_path=self._model_dir, file_name="params.txt"))
        self.logger = logging.getLogger("teacher_{0}".format(teacher_num))
        self._debug_dir = debug_dir
        if debug_dir and not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        self._snapshots = None
        self._episodes = None
        self._model = None
        self._replay = None
        self.total_lives = env_params['total_lives']
        self.init_teacher()

    def init_teacher(self):
        raise NotImplementedError

    def init_model(self):
        return init_atari_model(
            obs_space=tuple(self._params["env"]["obs_space"]),
            num_actions=self._params["env"]["num_actions"],
            hidden_size=self._params["train"]["hidden_size"],
            gpu=self._gpu_params["enabled"],
            gpu_id=self._gpu_params["id"])

    def _read_snapshot(self, number):
        """Used in subclass `SnapshotsTeacher` to load teacher snapshots.

        For progress nets, we load these into the _upper_ portion, except for
        the last (usually best) snapshot, which goes in the _lower_ part.

        Parameters
        ----------
        number: Zero-indexed integer index of matched snapshot for `theta_i`.
            We add +1 in the `read_snapshot` call.
        """
        message = "Loading snapshot {} (0-idx) w/avg reward {:.2f}, steps {}".format(
            number,
            self._snapshots[self._snapshots.target_field, number],
            self._snapshots["steps", number])
        message += " for the upper progress net"
        self.logger.debug(message)
        return read_snapshot(
            model_dir=self._model_dir, number_0_idx=number,
            gpu=self._gpu_params["enabled"], gpu_id=self._gpu_params["id"])

    def _read_episode(self, number):
        self.logger.debug(
            "Reading episode {} with {:.2f} rewards and {} steps".format(
                number,
                self._episodes[self._episodes.target_field, number],
                self._episodes["steps", number]))
        return read_episode(model_dir=self._model_dir, number_0_idx=number)

    def get_learner_weights(self, obs, acts):
        raise NotImplementedError

    def get_teacher_samples(self, batch_size, num_steps, steps):
        raise NotImplementedError

    def _create_replay(self):
        """Thought we were going to have ONE fixed XP replay buffer?

        I originally assumed for the teacher's buffer, we'd keep adding to that
        and removing older elements. It seems like instead we delete the prior
        one and start again from scratch. Fair enough ...
        """
        if self._replay is not None:
            del self._replay
        self._replay = ExperienceReplay(
            writer=self.writer,
            capacity=self._teacher_params["replay_size"],
            init_cap=0,
            frame_stack=self._env_params["frame_stack"],
            gamma=self._train_params["gamma"],
            tag="teacher_{0}".format(self.teacher_num),
            debug_dir=None)

    def _load_episode(self, number):
        _episode = self._read_episode(number=number)
        self._replay.add_episode(_episode)
