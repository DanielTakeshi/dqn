from dqn.teacher.teacher import Teacher
from torch.autograd import Variable
from dqn.utils.summary import IterativeSummary


class WeightTeacher(Teacher):
    def __init__(self, writer, model_name, teacher_num, train_params,
                 gpu_params, teacher_params, env_params, log_params, debug_dir):
        super().__init__(
            writer=writer, model_name=model_name, teacher_num=teacher_num,
            train_params=train_params, gpu_params=gpu_params,
            teacher_params=teacher_params, env_params=env_params,
            log_params=log_params, debug_dir=debug_dir)
        self.logger.info("Teacher {0} has been set to be a weight "
                         "teacher.".format(self.teacher_num))

    def init_teacher(self):
        self._snapshots = IterativeSummary(
            model_dir=self._model_dir, summary_type="snapshots",
            target_field="rew")
        self.update_snapshots_summary()
        self._model = self.init_model()
        _last_snapshot_weights = self._read_snapshot(
            number=self._snapshots.df_last_num)
        self._model.init_weight(weights=_last_snapshot_weights)
        self.logger.debug(
            "Snapshot {0} has been loaded to `_model` in teacher {1}."
            .format(self._snapshots.df_last_num, self.teacher_num))

    def _advantage_scores(self, obs, acts):
        assert self._model
        assert isinstance(obs, Variable)
        assert isinstance(acts, Variable)
        assert obs.shape[0] == acts.shape[0]

        _qs = self._model(obs)
        _qs = _qs - _qs.min(dim=1, keepdim=True)[0]
        return _qs.gather(1, acts.unsqueeze(-1)).squeeze(-1)

    def _credit_scores(self, obs):
        assert self._model
        assert isinstance(obs, Variable)

        _qs = self._model(obs)
        _qs = _qs - _qs.min(dim=1, keepdim=True)[0]
        return _qs.std(dim=1)

    def get_learner_weights(self, obs, acts):
        assert self._teacher_params["method"] == "credit" or \
            self._teacher_params["method"] == "adv"
        if self._teacher_params["method"] == "credit":
            return self._credit_scores(obs)
        else:
            return self._advantage_scores(obs, acts)

    def get_teacher_samples(self, batch_size, num_steps, steps):
        return None
