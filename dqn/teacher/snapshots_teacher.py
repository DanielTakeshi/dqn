from dqn.utils.train import init_atari_progress_model
from dqn.teacher.episodes_teacher import EpisodesMatchingTeacher
from dqn.utils.summary import IterativeSummary, MatchingSummary
from dqn.utils.io import write_dict, read_snapshot
from dqn.utils.schedules import LinearSchedule
from pprint import pformat
import os
from os.path import join
import sys
import json
import numpy as np
import time
import torch
from copy import deepcopy


class SnapshotsTeacher(EpisodesMatchingTeacher):

    def __init__(self, writer, model_name, teacher_num, train_params,
                 gpu_params, teacher_params, env_params, log_params,
                 debug_dir, opt_params):
        # We save these in the student's `teaching/teacher_k` sub-directory.
        self._debug_info = {
            "matchings": {}, "episodes": {},
        }
        self._debug_info_olap = {
            "overlap": {},
        }
        super().__init__(
            writer=writer, model_name=model_name, teacher_num=teacher_num,
            train_params=train_params, gpu_params=gpu_params,
            teacher_params=teacher_params, env_params=env_params,
            log_params=log_params, debug_dir=debug_dir, opt_params=opt_params)
        self.logger.info("Teacher {0} has been set to be a snapshot "
                         "teacher.".format(self.teacher_num))

        # beta in weighted importance sampling (TODO are we using??)
        steps = self._env_params["max_num_steps"] // self._train_params["train_freq_per_step"]
        self.beta_schedule = LinearSchedule(
            schedule_timesteps=steps,
            final_p=1.0,
            initial_p=self._teacher_params["init_wis_beta"])

        # Matching method, and load the pretrained overlap network if needed.
        match_methods = ["fixed_steps", "one_forward", "train_net"]
        self._match_method = self._teacher_params["overlap"]["match_method"]
        assert self._match_method in match_methods, self._match_method
        if self._match_method == "train_net" or self._match_method == "one_forward":
            self._load_pretrained_multiclass_net()

    @property
    def snapshots(self):
        return self._snapshots

    @property
    def episodes(self):
        return self._episodes

    @property
    def done_reading(self):
        assert self._snapshots is not None
        return self._snapshots.done

    @property
    def total_matchings(self):
        return len(self._debug_info["matchings"])

    @property
    def replay(self):
        assert self._replay is not None
        return self._replay

    def init_teacher(self):
        """Called during snapshot teacher creation.

        First create ProgressNet model, with progress scores. Then have
        summaries over the _episodes_ and _snapshots_, with (for example)
        various condense factors to reduce the amount of stuff to consider.

        HUGE NOTE: episodes summary matches w/older fields, `total_clip_rew`
        (formerly `total_rew`). Need to fix. But snapshots has correct key, I
        think ... but have to check that other code which matches it uses the
        true rewards, not the clipped ones ...
        """
        self._model = init_atari_progress_model(
            obs_space=self._params["env"]["obs_space"],
            num_actions=self._params["env"]["num_actions"],
            train_params=self._params["train"],
            teacher_params=self._teacher_params,
            gpu_params=self._gpu_params,
            max_num_steps=self._params["env"]["max_num_steps"],
            opt_params=self._opt_params)
        self._episodes = IterativeSummary(
            model_dir=self._model_dir,
            summary_type="episodes",
            target_field="raw_rew",
            lead_number=0)
        self._snapshots = MatchingSummary(
            model_dir=self._model_dir,
            summary_type="snapshots",
            target_field="true_rew_epis",
            lead_number=self._teacher_params["num_snapshot_ahead"],
            smooth_window=1)
        self._snapshots.cast_field_to_type("num_finished_epis", int)

        # self._read_and_load_last_snapshot()
        # I changed to best snapshot, should be fine, empirically it stops at best.
        self._read_and_load_best_snapshot()
        # self._progress_update(latest_reward=None, cum_steps=0, next_snapshot_num=1)
        # self.debug(latest_reward=None, cum_steps=0, start_num=1)

    def get_teacher_samples(self, batch_size, num_steps, steps):
        """Used for combo student/teacher minibatches.

        Teacher samples get sampled before the students. But, both use the
        replay buffer class, `ExperienceReplay`, w/different instances; the
        teacher's replay is not built until AFTER the student populates the
        replay buffer with its own initial samples.

        Here, `_replay.info` has keys: ['activations', 'progress_scores', 'p',
        '_progress_scores', 'progress_signs']) for `pong_snapshot_fast.json`.

        The `_wis` are importance sampling weights. We assume some probability
        distribution `_replay.info[p]`, from the PROGRESS SCORES. We sample the
        transitions according to them, and do the corrections here.
        """
        if self._replay is None:
            return None
        transitions, idx = self._replay.sample(
            batch_size, num_steps=num_steps, output_idx=True)
        _weights = transitions.weight
        if self._teacher_params["negative_correction"]:
            _weights *= self._replay.info["progress_signs"][idx]
        if "p" in self._replay.info:
            _wis = np.power(self._replay.info["p"][idx] * len(self._replay),
                            -self.beta_schedule.value(steps))
            _wis /= _wis.max()
            _weights *= _wis
        transitions.weight = _weights
        return transitions

    def _read_and_load_last_snapshot(self):
        """What Allen used. However, I think we want the BEST snapshot..."""
        number = self._snapshots.df_last_num
        _last_snapshot_weights = self._read_snapshot(number=number)
        self._model.load_lower_weight(model=_last_snapshot_weights)
        self.logger.debug("Snapshot {0} (the last one) has been loaded to "
                          "the lower part of progress net in teacher {1}".format(
                              number, self.teacher_num))

    def _read_and_load_best_snapshot(self):
        """I think we want this."""
        number = self._snapshots.df_idx_best_snapshot
        _best_snapshot_weights = self._read_snapshot(number=number)
        self._model.load_lower_weight(model=_best_snapshot_weights)
        self.logger.debug("Snapshot {0} (0-idx), BEST snapshot ({1} total), has "
            "been loaded to lower part of progress net in teacher {2}".format(
                number, len(self.snapshots), self.teacher_num))

    def _check_snapshots_direction(self, start_num, num_gaps=1):
        _snapshot_1 = read_snapshot(
            model_dir=self._model_dir, number=start_num,
            gpu=self._gpu_params["enabled"], gpu_id=self._gpu_params["id"])
        _snapshot_2 = read_snapshot(
            model_dir=self._model_dir, number=start_num + num_gaps,
            gpu=self._gpu_params["enabled"], gpu_id=self._gpu_params["id"])
        _last_snapshot = read_snapshot(
            model_dir=self._model_dir, number=self._snapshots.df_last_num,
            gpu=self._gpu_params["enabled"], gpu_id=self._gpu_params["id"])
        _progress_score = 0.0
        _vector1_norm = 0.0
        _vector2_norm = 0.0
        for name, params_1 in _snapshot_1.items():
            _vector2 = (_snapshot_2[name] - params_1).view(-1)
            _vector1 = (_last_snapshot[name] - params_1).view(-1)
            _progress_score += (_vector1 * _vector2).sum()
            _vector1_norm += (_vector1.norm()) ** 2
            _vector2_norm += (_vector2.norm()) ** 2
        return _progress_score, _vector1_norm, _vector2_norm

    def matching(self, rew: float, steps: int, agent=None):
        """Called from a 'teacher center' matching method to update snapshots.

        See if there's a matching snapshot for the student, using `_snapshots`
        object, a `utils.MatchingSummary`.  Most of the time, we won't need to
        update the match, so we `return` before the `_progress_update` call;
        it's because the agent is usually not improving enough to warrant
        constantly changing the matched snapshot.

        - We simply leave if we are done, if student's reward exceeds that of
          the best snapshot.
        - Automatically match if we have surpassed the initial exploration stage
          and are now starting the sampling from the replay buffer(s).
        - For current snapshot num, note that we explicitly added that 'number'
          field in `utils/summary.py`.

        Parameters
        ----------
        agent: dqn.agent.dqn.DQNTrainAgent
            Passing this here so we can refer to it, e.g., its replay buffer
            elements for later use in the overlap metric computations.

        Returns
        -------
        None, or the zpd_ss_num index. Doing this to make it easier to detect
        in external code when we are able to do some matching. Though we might
        as well remove it if we are not using it ...
        """
        if self.done_reading:
            return None
        # The current and matched should often be the same.
        current_ss_num = self.snapshots.last_match_idx

        if self.snapshots.has_exceeded_last_match(rew):
            # Consider different methods for matching to get `zpd_ss_num`.
            start_t = time.time()
            if self._match_method == "fixed_steps":
                matched_ss_num, zpd_ss_num = self.snapshots.next(rew,
                        t_params=self._teacher_params)
            elif self._match_method in ["one_forward", "train_net"]:
                matched_ss_num, zpd_ss_num, olap_info = self.snapshots.next_overlap(
                        rew, agent,
                        t_params=self._teacher_params,
                        match_method=self._match_method,
                        overlap_ensemble=self.overlap_nets,
                        overlap_train_args=self.overlap_train_args,
                        gpu_params=self._gpu_params,
                        total_data_elements=self.total_data_elements,
                        info_per_class=self.info_per_class,
                        pt_data_dir=self.pt_data_dir,
                        student_logs=self._log_params)
                self._debug_olap(rew, steps, current_ss_num, olap_info)
            else:
                raise ValueError(self._match_method)

            # Get samples from `zpd_ss_num` in teacher's replay buffer.
            self._progress_update(rew, steps, current_ss_num, matched_ss_num, zpd_ss_num)
            self.debug(rew, steps, current_ss_num)
            zpd_t = (time.time() - start_t) / 60.
            self.logger.debug('Identified the ZPD in {:.1f} mins'.format(zpd_t))
            return zpd_ss_num
        else:
            self.logger.info(
                "Agent has not exceeded current snapshot. "
                "Current snapshot: {}, len(self._replay): {}.".format(
                    current_ss_num, len(self._replay)))
            return None

    def _progress_update(self,
                         latest_reward,
                         cum_steps,
                         previous_ss_num,
                         matched_ss_num,
                         zpd_ss_num):
        """Only called when we need to update matched snapshots.

        Load ZPD teacher (not the matched teacher!) transitions into replay
        buffer, but sometimes we might be re-matching to the same ZPD teacher.

        Called by `self.matching`, which checks if it is necessary to even
        update the next snapshot --- usually we don't.  Calls superclass'
        `load_transitions` which prints the 'By ZPD ...' logging message.

        If making progress scores, we use the *matched* (not ZPD) snapshot for
        the progress scores. If we want to sample uniformly from the replay
        buffer, then we don't compute scores. That forces the replay info to be
        `{}` which means we don't weigh according to progress scores.

        Parameters
        ----------
        The previous, matched, and zpd snapshot numbers, 0-indexed. Use the zpd
        one for loading transitions and the matched for progress score.
        """
        self.load_transitions(latest_reward, cum_steps, previous_ss_num,
                              matched_ss_num, zpd_ss_num)
        if self._teacher_params["temperature"] > 0 and not self._teacher_params["teacher_samples_uniform"]:
            self._replay.info = self.progress_score(matched_ss_num, self._replay)
        else:
            self._replay.info = {}

    def load_transitions(self,
                         latest_reward,
                         cum_steps,
                         previous_ss_num,
                         matched_ss_num,
                         zpd_ss_num):
        """Load zpd_ss_num's teacher transitions into replay buffer.

        Call cycle: matching() -> _progress_update() -> load_transitions().

        Load TEACHER transitions from its saved pickle files, about a window
        which we detect via cum_steps. We detect the number of episodes
        finished to get the episode number, which we then map to lives (via the
        train_v02 summary file, as of May 2019) and then load from the nearest
        pickle files.

        Parameters
        ----------
        The previous, matched, and zpd snapshot numbers, 0-indexed. Use the zpd
        one for loading transitions and the matched for progress score.
        """
        _episode_num = self._snapshots["num_finished_epis", zpd_ss_num]

        # Both zero-indexed, refer into condensed_df, updated in `next()`.
        match_idx = self._snapshots.last_match_idx
        zpd_idx = self._snapshots.last_zpd_idx

        self.logger.info(
            "Agent ({0:.2f} rew, {1} cum steps) has matched to snapshot {2} "
            "({3:.2f} rew, {4} cum steps), prior match was {5} ({6:.2f} rew, "
            "{7} cum steps). The ZPD snapshot is {8} ({9:.2f} rew, "
            "{10} cum steps). [Snapshot numbers here are 0-indexed]".format(
                latest_reward,
                cum_steps,
                self._snapshots["number", match_idx],
                self._snapshots[self._snapshots.target_field, match_idx],
                self._snapshots["steps", match_idx],
                previous_ss_num,
                self._snapshots[self._snapshots.target_field, previous_ss_num],
                self._snapshots["steps", previous_ss_num],
                self._snapshots["number", zpd_idx],
                self._snapshots[self._snapshots.target_field, zpd_idx],
                self._snapshots["steps", zpd_idx]))
        self.logger.info("  zpd_idx, zpd_ss_num: {}, {}".format(zpd_idx, zpd_ss_num))
        self.load_episodes_in_window(_episode_num)

    def generate_progress_score(self, transitions):
        """Called via `_progress_score` when we need to update target.

        See `AtariProgressNet.progress_scores(...)` for details, there's a lot.
        Again, `transitions` is the entire teacher replay buffer.

        Returns
        -------
        _output_info: Used to _assign_ to teacher's `replay.info`, which has
            progress scores (plus a few other progress-related statistics).
        """
        _output_info = self._model.progress_scores(
            transitions=transitions,
            num_steps=self._train_params["num_steps"],
            activation=True)
        return _output_info

    def progress_score(self, matched_num, transitions):
        """Called via `_progress_update` when we need to update target.

        Loads the UPPER part of the network with the matched snapshot. The
        lower part already has the best snapshot loaded. Then get progress
        scores based on that weight and samples from the ZPD snapshot.

        Parameters
        ----------
        matched_num: (Condensed) integer index of the matched snapshot num that
            we use for `theta_i` in our notation. EDIT: wait I don't think it's
            condensed ... I think the raw index (starting from 1).
        transitions: The _entire_ teacher replay buffer based on the ZPD
            snapshot, which is NOT (in general) the same as the one represented
            by `matched_num`.
        """
        self.logger.info(
            "loading snapshot num {} in UPPER progress net".format(matched_num))
        matched_snapshot = self._read_snapshot(matched_num)
        self._model.load_upper_weight(matched_snapshot)
        return self.generate_progress_score(transitions)

    def progress_step(self, steps, transitions):
        self._model.step(transitions=transitions, steps=steps)

    def _debug_olap(self, latest_reward, cum_steps, prev_ss_num, olap_info):
        """See student's file: `[...]/teaching/teacher_k/overlap.txt`
        """
        _cm = self.total_matchings  # current matched
        match_idx = self._snapshots.last_match_idx
        zpd_idx = self._snapshots.last_zpd_idx
        self._debug_info_olap["overlap"][_cm] = {
            "learner_reward": latest_reward,
            "learner_cum_steps": int(cum_steps),
            "previous_ss_num": int(prev_ss_num),
            "previous_ss_reward": self._snapshots[self._snapshots.target_field, prev_ss_num],
            "previous_ss_steps": self._snapshots["steps", prev_ss_num],
            "matched_ss_num": self._snapshots["number", match_idx],
            "matched_ss_reward": self._snapshots[self._snapshots.target_field, match_idx],
            "matched_ss_steps": self._snapshots["steps", match_idx],
            "zpd_ss_num": self._snapshots["number", zpd_idx],
            "zpd_ss_reward": self._snapshots[self._snapshots.target_field, zpd_idx],
            "zpd_ss_steps": self._snapshots["steps", zpd_idx],
            "last_match_idx": match_idx,
            "last_zpd_idx": zpd_idx,
        }
        if 'avg_class_prob' in olap_info:
            self._debug_info_olap["overlap"][_cm]['avg_class_prob'] = \
                    olap_info['avg_class_prob'].tolist()
        if 'avg_olap_acc_mean' in olap_info:
            self._debug_info_olap["overlap"][_cm]['avg_olap_acc_mean'] = \
                    olap_info['avg_olap_acc_mean'].tolist()
            self._debug_info_olap["overlap"][_cm]['avg_olap_acc_std'] = \
                    olap_info['avg_olap_acc_std'].tolist()
            self._debug_info_olap["overlap"][_cm]['avg_olap_min_mean'] = \
                    olap_info['avg_olap_min_mean'].tolist()
            self._debug_info_olap["overlap"][_cm]['avg_olap_min_std'] = \
                    olap_info['avg_olap_min_std'].tolist()
        write_dict(self._debug_info_olap, self._debug_dir, "overlap")

    def debug(self, latest_reward, cum_steps, prev_ss_num):
        """See student's file: `[...]/teaching/teacher_k/summary.txt`

        We can use this to plot and inspect the matching process. It saves an
        equal amount of 'matchings' and 'episodes' data points. Also, don't
        worry, `cum_steps` is same as the thing I call training `steps`.
        """
        _current_matched = self.total_matchings
        match_idx = self._snapshots.last_match_idx
        zpd_idx = self._snapshots.last_zpd_idx

        self._debug_info["matchings"][_current_matched] = {
            "learner_reward": latest_reward,
            "learner_cum_steps": int(cum_steps),
            "previous_ss_num": int(prev_ss_num),
            "previous_ss_reward": self._snapshots[self._snapshots.target_field, prev_ss_num],
            "previous_ss_steps": self._snapshots["steps", prev_ss_num],
            "matched_ss_num": self._snapshots["number", match_idx],
            "matched_ss_reward": self._snapshots[self._snapshots.target_field, match_idx],
            "matched_ss_steps": self._snapshots["steps", match_idx],
            "zpd_ss_num": self._snapshots["number", zpd_idx],
            "zpd_ss_reward": self._snapshots[self._snapshots.target_field, zpd_idx],
            "zpd_ss_steps": self._snapshots["steps", zpd_idx],
            "last_match_idx": match_idx,
            "last_zpd_idx": zpd_idx,
        }
        self.writer.add_scalar(
            tag="teacher_{0}/matched_reward".format(self.teacher_num),
            scalar_value=self._snapshots[self._snapshots.target_field, zpd_idx],
            global_step=_current_matched)
        self.logger.info(pformat(self._debug_info["matchings"][_current_matched]))
        self._debug_info["episodes"][_current_matched] = self._replay.debug(_current_matched)
        self.writer.add_scalar(
            tag="teacher_{0}/matchings".format(self.teacher_num),
            scalar_value=self._teacher_params["temperature"],
            global_step=_current_matched)
        self.logger.debug(pformat(self._debug_info["episodes"][_current_matched]))
        write_dict(self._debug_info, self._debug_dir, "summary")

    def _load_pretrained_multiclass_net(self):
        """Called during the class init; loads pretrained multiclass network.

        We trained beforehand using the `distances` package.  Load the most
        recent set of trained models (or a specific model if specified) and
        then initialize the ensemble of networks with those weights.

        Handles both match methods of doing a single forward pass (through the
        ensemble) or training the ensemble starting from these loaded weights
        (i.e., fine tuned).  If it's the latter, we need a way to keep the
        network re-initialized in the same way.

        See `settings/README.md` for additional documentation. Also:
            https://pytorch.org/tutorials/beginner/saving_loading_models.html

        We need to import net classes from `distances.nets`, and know the
        training arguments, to make the networks. If the weights do not match
        the network architecture, we correctly get a runtime error.

        Assigns many class variables to this teacher, making it easy to refer
        to later in the training loop later in utils/summary.
        """
        try:
            from distances.nets import MulticlassNet
            from distances.utils import find_distances_multi_files
            from distances import utils_dataloaders as UD
        except ImportError as e:
            self.logger.info("Cannot import `distances`: {}".format(e))

        # (0) All teachers must have these two directories present!
        dist_dir = join(self._model_dir, 'distances_multi')
        torch_dir = join(self._model_dir, 'distances_pytorch_data')
        pretrained_m = self._teacher_params['overlap']['pretrained_model']
        if pretrained_m == "":
            pretrained_m = None
        self.logger.debug("Loading pretrained multiclass for overlap, from {}, "
                "pretrained_m: {}".format(dist_dir, pretrained_m))
        assert os.path.exists(dist_dir), "Doesn't exist:\n\t{}".format(dist_dir)
        assert os.path.exists(torch_dir), "Doesn't exist:\n\t{}".format(torch_dir)

        # (1) Load from `distances_multi` along with some sanity checks.
        files = find_distances_multi_files(parent_dir=dist_dir, name=pretrained_m)
        pt_files = sorted([x for x in files if '.pt' in x and 'fold' in x])
        train_args_pth = [x for x in files if '.json' in x]
        assert len(train_args_pth) == 1, train_args_pth
        train_args_pth = train_args_pth[0]
        with open(join(dist_dir, train_args_pth), 'r') as fh:
            train_args = json.load(fh)
        num_folds = train_args['num_folds']
        assert len(pt_files) == num_folds, \
                "pt files & folds, {} vs {}".format(len(pt_files), num_folds)
        dropout_p = train_args['dropout_p']
        if not train_args['apply_dropout']:
            dropout_p = 0.0
        assert train_args['agent'] in self._teacher_params['models'], \
                "{} not in {}".format(train_args['agent'], self._teacher_params['models'])

        # (2) Load from `distances_pytorch_data` along with some sanity checks.
        dl = UD.load_from_dataloader(path=torch_dir)
        info_results = dl['info_results']
        eps_df = dl['eps_df']
        snp_df = dl['snp_df']
        num_total = dl['num_total']
        info_per_class = dl['info_per_class']
        num_classes = len(info_results['models_to_test'])
        self.logger.debug("Loaded from {}".format(torch_dir))
        self.logger.debug("num_total {}, num_classes {}".format(num_total, num_classes))

        # (3) Now actually create and load the *models* which we trained beforehand.
        self.overlap_nets = []
        for pytorch_name in pt_files:
            model_full_path = join(dist_dir, pytorch_name)
            net = MulticlassNet(obs_space=(4,84,84),
                                num_classes=num_classes,
                                dropout=dropout_p)
            net.load_state_dict(torch.load(model_full_path))
            net.cuda()
            # Subtle point: if we're not (re-)training, we must get in eval mode.
            if self._match_method == "one_forward":
                net.eval()
            self.overlap_nets.append(net)

        self.logger.debug("Finished loading pre-trained ensemble for overlaps "
                "(ensemble size {})".format(len(self.overlap_nets)))

        # (4) Assign a bunch of class variables for future reference.
        self.total_data_elements = num_total
        self.info_per_class = info_per_class
        train_args['num_classes'] = num_classes
        self.overlap_train_args = train_args
        self.pt_data_dir = torch_dir

class SnapshotsReviewer(SnapshotsTeacher):
    """D: ??? I don't think we use this in `atari.py`.
    """
    def __init__(self, writer, model_name, learner_id, train_params,
                 gpu_params, teacher_params, env_params, log_params, debug_dir,
                 exp_queue, opt_params):
        super().__init__(
            writer=writer, model_name=model_name, teacher_num=learner_id,
            train_params=train_params, gpu_params=gpu_params,
            teacher_params=teacher_params, env_params=env_params,
            log_params=log_params, debug_dir=debug_dir, opt_params=opt_params)
        self._exp_queue = exp_queue
        self._upload = 0
        self._upload_time = 0

    def timer(self):
        return {
            "upload_amount": self._upload,
            "upload_time": self._upload_time
        }

    def reset_timer(self):
        self._upload = 0
        self._upload_time = 0

    def init_teacher(self):
        self._model = init_atari_progress_model(
            obs_space=self._params["env"]["obs_space"],
            num_actions=self._params["env"]["num_actions"],
            train_params=self._params["train"],
            teacher_params=self._teacher_params,
            gpu_params=self._gpu_params,
            max_num_steps=self._params["env"]["max_num_steps"],
            opt_params=self._opt_params)
        _new_path = join(self._model_dir, "learner_{0}".format(self.teacher_num))
        self._model_dir = _new_path
        self._episodes = IterativeSummary(
            model_dir=self._model_dir, summary_type="episodes",
            target_field="total_rew",
            lead_number=0, training=True)
        self._snapshots = MatchingSummary(
            model_dir=self._model_dir, summary_type="snapshots",
            target_field="rew",
            lead_number=self._teacher_params["num_snapshot_ahead"],
            smooth_window=1)

    def matching(self, latest_reward, steps):
        assert len(self.snapshots) >= 2
        assert len(self.episodes) >= 1

        self._read_and_load_last_snapshot()
        _target_old_snapshot = max(
            self._snapshots.df_last_num - self._teacher_params[
                "num_snapshot_ahead"], 1)
        self._progress_update(latest_reward=latest_reward,
                              cum_steps=steps,
                              next_snapshot_num=_target_old_snapshot,
                              last_snapshot_num=1)
        self.debug(latest_reward=latest_reward, cum_steps=steps,
                   start_num=_target_old_snapshot)
        self._add_transitions_to_queue()

    def get_one_sample(self):
        _transition, idx = self._replay.sample_one(
            num_steps=self._train_params["num_steps"], output_idx=True)
        if self._teacher_params["negative_correction"]:
            _transition.weight *= self._replay.info["progress_signs"][idx]
        return _transition

    def _add_transitions_to_queue(self):
        current_queue_len = self._exp_queue.qsize()
        for i in range(self._teacher_params["num_shared_exp"]):
            if current_queue_len > 0:
                self._exp_queue.get()
                current_queue_len -= 1
            assert not self._exp_queue.full()
            _start_timer = time.time()
            self._exp_queue.put(self.get_one_sample())
            self._upload_time += time.time() - _start_timer
            self._upload += 1


class SelfReviewer:
    def __init__(self, obs_space, num_actions, train_params, gpu_params,
                 teacher_params, total_teachers, opt_params, max_num_steps):
        self._train_params = train_params
        self._total_teachers = total_teachers
        self._model = init_atari_progress_model(
            obs_space=obs_space,
            num_actions=num_actions,
            train_params=train_params,
            teacher_params=teacher_params,
            gpu_params=gpu_params,
            max_num_steps=max_num_steps,
            opt_params=opt_params)

    def examine(self, net, target, transitions):
        assert isinstance(transitions, dict)
        self._model.load_weight(
            upper=target.state_dict(),
            lower=net.state_dict())
        output_scores = np.zeros(self._total_teachers)
        num_transitions = np.zeros(self._total_teachers)
        for teacher_id, transition in transitions.items():
            transition.stack()
            num_transitions[teacher_id] = len(transition)
            info = self._model.progress_scores(
                transitions=transition,
                num_steps=self._train_params["num_steps"],
                activation=False)
            output_scores[teacher_id] = info["progress_scores"].mean()
        return output_scores, num_transitions
