import os
import sys
import copy
import time
import logging
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np
import pandas as pd
from dqn.utils.variables import (create_tensor, cuda)
from os.path import join
pd.set_option('display.max_rows', 500)
from collections import defaultdict
# Help with debugging the forward passes.
torch.set_printoptions(precision=3, edgeitems=10, linewidth=200)
np.set_printoptions(suppress=True, precision=4, edgeitems=10, linewidth=300)


class Summary:
    """Used for teachers to manage episode and snapshot matching/selection.
    """
    def __init__(self,
                 model_dir: str,
                 summary_type: str,
                 target_field: str,
                 training: bool = True):
        assert summary_type in ["episodes", "snapshots"]
        self.target_field = target_field
        self._model_dir = model_dir
        self._summary_type = summary_type
        self._training = training
        self._data_frame = self._read_summary(model_dir, summary_type, training)

    def cast_field_to_type(self, field: str, dtype: type):
        self._data_frame[field] = self._data_frame[field].astype(dtype)

    def _read_summary(self, model_dir: str, summary_type: str, training: bool) -> pd.DataFrame:
        """Reads a summary file from given model directory based on the summary type.

        :model_dir: (str) Directory from which to load model information
        :summary_type: (str) one of 'episodes' or 'snapshots'
        :training: (bool) whether to load the training summary or testing summary

        :returns: pandas DataFrame read from the summary file
        """
        assert summary_type in ["episodes", "snapshots"], summary_type
        summary_dir = os.path.join(model_dir, summary_type)
        if summary_type == "episodes":
            summary_file = "{}_summary_true_v02.txt".format(
                "training" if training else "testing")
        else:
            summary_file = "snapshots_summary.txt"
        data_frame = pd.read_json(os.path.join(summary_dir, summary_file), orient='index')

        # Everything in our code should be 0 indexed, except for saving of pkl files.
        # But earlier, I had been using one-indexing. Argh ... let's standardize.
        assert data_frame.index[0] in [0, 1], data_frame.index[0]
        if data_frame.index[0] == 1:
            data_frame.index -= 1

        # map to python type instead of numpy type
        data_frame.index = data_frame.index.map(int)
        data_frame.sort_index(inplace=True)
        print(data_frame)
        return data_frame

    @property
    def _ignore_last(self):
        raise NotImplementedError

    @property
    def condensed_last_idx(self):
        return len(self) - 1 - int(self._ignore_last)

    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def _condensed_contains_idx(self, idx):
        return idx in self._data_frame.index

    def __contains__(self, idx: int) -> bool:
        return idx in self._data_frame.index

    def __getitem__(self, key: Tuple[str, int]):
        field, idx = key
        if field == "number":
            return idx
        # use item to return python type instead of numpy type
        return self._data_frame[field][idx].item()


class IterativeSummary(Summary):

    def __init__(self,
                 model_dir: str,
                 summary_type: str,
                 target_field: str,
                 lead_number: int = 0,
                 training: bool = True):
        """For matching episodes."""
        super().__init__(
            model_dir=model_dir, summary_type=summary_type,
            target_field=target_field, training=training)
        self._current = lead_number

    @property
    def _ignore_last(self):
        return False

    @property
    def done(self):
        return self._next_read_num not in self

    def next(self):
        if not self.done:
            _current_read_num = self._next_read_num
            self._next_read_num += 1
            return _current_read_num
        else:
            return None


class MatchingSummary(Summary):

    def __init__(self,
                 model_dir: str,
                 summary_type: str,
                 target_field: str,
                 lead_number: int = 0,
                 training: bool = True,
                 smooth_window: int = 1):
        """For matching snapshots.

        last_match_idx: index into df to find current/matched snapshot
        last_zpd_idx: index into df to find ZPD snapshot_next

        Both are zero-indexed.
        """
        self._lead_number = lead_number
        self.last_match_idx = 0
        self.last_zpd_idx = 0
        self._has_matched_to_something = False
        assert smooth_window >= 1
        self._smooth_window = smooth_window
        self._are_we_done = False
        self.logger = logging.getLogger("match_summary")
        super().__init__(model_dir=model_dir, summary_type=summary_type,
                         target_field=target_field, training=training)

    def _read_summary(self,
                      model_dir: str,
                      summary_type: str,
                      training: bool) -> pd.DataFrame:
        """We can optionally smooth the target after reading it.

        BUT I would not do this ... we already smooth by taking the average of
        the past 100 episode rewards (that's what the self.target_field key means).
        Fortunately `smooth_window=1` should mean this does nothing.
        """
        data_frame = super()._read_summary(model_dir, summary_type, training)
        data_frame = data_frame.rolling(window=self._smooth_window, min_periods=1, center=True).mean()
        return data_frame

    @property
    def df_idx_best_snapshot(self):
        """The snapshot index (0-indexed) with highest target_field value."""
        return self._data_frame[self.target_field].idxmax()

    @property
    def _ignore_last(self):
        """AH! We ignore the last snapshot!

        That explains why we don't match to the actual last snapshot. Though in
        our case it's probably OK to nto worry about it. The best, not last,
        snapshot is used for the progress score network. And even if earlier, we
        do match to that best one, the progress scores should just be the same,
        all zeros which the softmax turns to a uniform distribution ... ?
        """
        return True

    @property
    def done(self):
        """Used to determine if we're done matching wrt this particular teacher.

        `MultiTeacherTeachingCenter.matching` --> `SnapshotsTeacher.done_reading`

        Previously, based on: `self.last_zpd_idx >= self.condensed_last_idx`.

        But suppose we have a large ZPD interval. Then we hit the last ZPD idx
        at a time when our actual _matched_ snapshot is much earlier. Our
        progress scores thus would stay fixed wrt the matched snapshot, and we'd
        apply the other decay schedule from the teacher to eventually use no
        teacher samples. It makes more sense to keep matching (but with the same
        ZPD teacher) until the matched snapshot is basically the best one?
        """
        return self._are_we_done

    def has_exceeded_last_match(self, value: float) -> bool:
        """True if agent exceeds previous matched snapshot (or hasn't matched at all).

        :value: (float) average current value
        :returns: (bool) if current value greater than previous matched value
        """
        return (not self._has_matched_to_something) or \
                value > self[self.target_field, self.last_match_idx]

    def next(self, value: float, *, t_params) -> Tuple[int, int]:
        """From SnapshotsTeacher, find next matched snapshot given current rew.

        The method is called often, after each episode or so, but usually we
        should not need to keep re-matching. Our algorithm is that the agent
        gets matched to a snapshot *after* it with higher reward, but with the
        *closest* such high reward. THEN, we do a ZPD where we search a fixed
        number of snapshots ahead.

        Roshan: I replaced the while loop with a broadcasted subtract of the
        value, followed by some filtering and an idxmin. This is cleaner and
        faster than the old code. Daniel: doing diff[diff >= 0] will preserve
        indices for the snapshots w/higher value than agent reward, clever. But
        we need to protect against having an empty sequence!

        Parameters
        ----------
        value: Float representing past average (usually via window of 100)
            rewards by the (learner) agent.

        Returns
        -------
        (match_idx, zpd_idx): A simple tuple from the condensed table
            representing the snapshot indices to match and ZPD, respectively.
            We actually don't need to return these as they are set to class
            variables; it's mostly for debugging.

        TODO (Roshan):
            if self.done is true, this returns None. Instead it should probably
            do something like raise a StopIteration, which would make this
            more similar to an iterator.

            I've been trying to figure out whether this makes more sense as a
            genuine iterator, but because the next value depends on a user
            input, it probably doesn't. I suppose we could write this as a
            generator and have the user send a value to the generator, but
            is that better?
        """
        if self.done:
            raise StopIteration("At end of summary, can't match any more values")
        diff = self._data_frame[self.target_field] - value
        diff = diff[self.last_match_idx:]
        if self._ignore_last:
            diff = diff[:-1]

        # Daniel: from Roshan's code, I think this is where we should check if
        # done. It's empty when agent exceeds value of all possible snapshots.
        # Outside of the init, we call this if last match has been _exceeded_.
        if len(diff[diff >= 0]) == 0:
            self._are_we_done = True
            self.logger.info("self._are_we_done=True, since len(diff[diff>=0])=0.")
            closest_idx = self.last_match_idx
        else:
            closest_idx = diff[diff >= 0].idxmin().item()
            closest_diff = diff[closest_idx].item()
            closest_rew = self[self.target_field, closest_idx]
            self.logger.debug("closest idx (0-idx), diff, s-rew: {}, {:.5f}, {:.5f}".format(
                closest_idx, closest_diff, closest_rew))

        # Set ZPD as a fixed number ('lead') ahead, and don't exceed maximum index.
        # July 18, 2019: to fairly compare with ZPD we should only ZPD with
        # snapshots that are the same as those we'd consider for the overlap.
        # Thus, zpd_idx must be 'mapped' to the nearest ZPD we consider.
        condense_f = t_params['condense_freq']
        self.logger.debug("last_match_idx: {}  (zero-indexed)".format(self.last_match_idx))
        self.logger.debug("learner reward: {:.2f}".format(value))
        zpd_idx = closest_idx + (self._lead_number * condense_f)
        idx_candidates = condense_f * (1 + np.arange(self.condensed_last_idx))
        # Remove anything that exceeds len(self)
        _max_idx = 0
        for idx in range(len(idx_candidates)):
            if idx_candidates[idx] > len(self)-1:
                _max_idx = idx
                break
        idx_candidates = idx_candidates[:_max_idx]
        # Difference btwn candidates and zpd, the get the integer index of it
        zpd_idx = int( idx_candidates[ np.abs(idx_candidates - zpd_idx).argmin() ] )

        self.last_match_idx = closest_idx
        self.last_zpd_idx = min(zpd_idx, len(self) - 1)
        self._has_matched_to_something = True

        # Handle the 'best ahead' case, https://github.com/CannyLab/dqn/issues/65.
        if self._lead_number == -1:
            filtered = self._data_frame[self.target_field][idx_candidates]
            self.logger.info("NOTE! _lead_number=-1 so fix ZPD as best (condensed) teacher.")
            self.last_zpd_idx = int(filtered.idxmax())

        # Handle the random case, but we need to filter for only better candidates.
        if self._lead_number == -2:
            random_candidates = idx_candidates[idx_candidates > closest_idx]
            self.logger.info("NOTE! _lead_number=-2 so we RANDOMIZE the ZPD.")
            self.logger.info("There are {} ZPD candidates: {}".format(
                    len(random_candidates), random_candidates))
            if len(random_candidates) == 0:
                # Just get the last (valid) one, 115 in our setup.
                self.last_zpd_idx = int(idx_candidates[-1])
            else:
                rand_idx = np.random.randint( len(random_candidates) )
                self.last_zpd_idx = int(random_candidates[rand_idx])

        return self.last_match_idx, self.last_zpd_idx

    def next_overlap(self, value, agent, *, t_params, match_method,
            overlap_ensemble, overlap_train_args, gpu_params, total_data_elements,
            info_per_class, pt_data_dir, student_logs) -> Tuple[int, int]:
        """From SnapshotsTeacher, find next matched snapshot using forward pass.

        Similar to `self.next` method. The lead number is irrelevant here.
        There seem to be two reasonable options. One for the forward pass:

        (0) Requires the `distance_multi` directory to have the trained network
            from classifying N different snapshots, usually around 20-ish.
        (1) Load the last several thousand states from student's buffer. Should
            be done in a similar manner as we did for the overlap training
            (with ignore_k applied, etc.).
        (2) Pass all the student states through that loaded network, to get the
            probabilities of the states as belonging in teacher classes.
        (3) Average probabilities across all student states, and select the
            corresponding snapshot with probability closest to our target.
            Should restrict to the snapshots with higher reward, though.

        (See: https://github.com/CannyLab/dqn/issues/51)

        and another involving training a network:

        (0) Requires the `distance_multi` directory to have the trained network
            from classifying N different snapshots, usually around 20-ish.
        (1) Load the last several thousand states from student's buffer. Should
            be done in a similar manner as we did for the overlap training
            (with ignore_k applied, etc.).
        (2) But unlike the single forward pass case, we also need to obtain a
            similar set of data for EACH of the teacher snapshots. Thus, we
            need a similar amount of data for each of the N teachers plus the
            student (the (N+1)-th class).
        (3) Load that trained network. Recall that it has a pre-trained set of
            convolutional filters, followed by several layers (that we
            trained!), and then the last layer which maps to N classes. Change
            the network so that we map to N+1 classes, i.e., just change the
            very last layer.
        (4) Then train! Keep the convolutional filters fixed (or maybe not),
            but also initialize the next layers from the pretrained network so
            that we are fine-tuning (or maybe just fix) from there. I think
            it's easiest to initialize the last layer from scratch, but we
            could try and see if we can init from the pretrained weights and
            figure out components in the weight matrix that should be init'd
            randomly for the student's data. But regardless, at least the fine
            tuning here should make training relatively quick.
        (5) With the trained network, we compute overlaps like we have been
            doing in the distances. Whichever snapshot has the closest overlap
            with the student (based on our tuned target) is chosen. Again,
            restrict to snapshots with higher reward.

        Returns the usual (match idx, zpd idx) pair, plus an extra dictionary
        for debugging this particular teacher.
        """
        if self.done:
            raise StopIteration("At end of summary, can't match any more values")
        diff = self._data_frame[self.target_field] - value
        diff = diff[self.last_match_idx:]
        if self._ignore_last:
            diff = diff[:-1]

        # I think we do two parts. First, we find the closest idx here. THEN get
        # ZPD index. Probably best for fair comparisons among 'ZPD-finding' methods.
        if len(diff[diff >= 0]) == 0:
            self._are_we_done = True
            self.logger.info("self._are_we_done=True, since len(diff[diff>=0])=0.")
            closest_idx = self.last_match_idx
        else:
            closest_idx = diff[diff >= 0].idxmin().item()
            closest_diff = diff[closest_idx].item()
            closest_rew = self[self.target_field, closest_idx]
            self.logger.debug("closest idx (0-idx), diff, s-rew: {}, {:.5f}, {:.5f}".format(
                closest_idx, closest_diff, closest_rew))

        # Hard-coded thresholds, and the args from overlap training run.
        prob_target = t_params["overlap"]["prob_target"]
        overlap_target = t_params["overlap"]["overlap_target"]
        l2reg = overlap_train_args["l2reg"]
        lrate = overlap_train_args["lrate"]
        ignore_k = overlap_train_args["ignore_k"]
        num_states = overlap_train_args["steps_per_class"]
        condense_f = overlap_train_args["condense_freq"]
        num_models = overlap_train_args["num_classes"]
        num_folds = overlap_train_args["num_folds"]
        batch_size = 256
        nb_parallel_workers = 8
        nb_epochs = 2

        # Get most recent episodes/states from the student buffer. Go from end of
        # the episode to the beginning, in case we have to stop data
        # collection. Depends on which method we're using.
        learner_buf = agent._replay.buffer

        if match_method == "one_forward":
            # ---------------------------------------------------------------- #
            # Using states, do single forward pass through the ensemble network.
            # We have ensemble predictions of size (N,classes), but how to use?
            # A simple way is to just take the average across all N items.
            # See June 11 hand written notes, idx_candidates are zero-indexed.
            # We assume we can never possibly match the 0-indexed snapshot.
            # ---------------------------------------------------------------- #
            states = []
            done = False
            for ep_idx in range(len(learner_buf) - 1, -1, -1):
                if done:
                    break
                episode_states = learner_buf[ep_idx].states
                episode_length = learner_buf[ep_idx].length
                if len(episode_states) <= ignore_k:
                    continue
                for s_idx in range(len(episode_states) - 4, ignore_k, -1):
                    obs = episode_states[s_idx : s_idx + 4]
                    assert obs.shape == (4, 84, 84), obs.shape
                    states.append(obs)
                    if len(states) >= num_states:
                        self.logger.debug("breaking, len(states): {}".format(len(states)))
                        done = True
                        break
                self.logger.debug("epidx {} (0-idx'd) s/l len {}/{}, states len {}".format(
                        ep_idx, len(episode_states), episode_length, len(states)))

            # Pass the states through the networks and compute desired stuff.
            states = np.array(states)
            self.logger.info("finished forming learner data in array: {}".format(states.shape))

            ensemble_preds = None
            xs = create_tensor(states,
                               gpu=gpu_params['enabled'],
                               gpu_id=gpu_params['id'],
                               gpu_async=gpu_params['async'],
                               convert_func='numpy')
            # Get net outputs, normalize probs, and get ensemble predictions.
            for net in overlap_ensemble:
                with torch.set_grad_enabled(False):
                    preds_BC = net(xs)
                    vals_B, _ = torch.max(preds_BC, dim=1, keepdim=True)
                    preds_BC = torch.exp(preds_BC - vals_B)  # numerical stability
                    probs_BC = preds_BC / torch.sum(preds_BC, dim=1, keepdim=True)
                    probs_BC_np = probs_BC.cpu().numpy()
                    if ensemble_preds is None:
                        ensemble_preds = probs_BC_np
                    else:
                        ensemble_preds += probs_BC_np
            ensemble_preds /= len(overlap_ensemble)
            assert len(overlap_ensemble) == num_folds, len(overlap_ensemble)
            assert ensemble_preds.shape[0] == states.shape[0], ensemble_preds.shape

            avg_class_prob = np.mean(ensemble_preds, axis=0)
            differences = np.abs(prob_target - avg_class_prob)
            idx_candidates = condense_f * (1 + np.arange(num_models))
            diff_idx = [(x,y) for (x,y) in zip(differences,idx_candidates)]
            diff_idx = sorted(diff_idx, key=lambda x:x[0])

            # Create zpd_idx in case the loop might not assign to it.
            zpd_idx = None
            for dd,snap_idx in diff_idx:
                # Ignore snapshots with lower index than current matched one.
                if snap_idx <= closest_idx:
                    continue
                # Ignore snapshots with lower reward than current value.
                if diff[snap_idx] < 0:
                    continue
                # Otherwise we're at the closest snapshot idx satisfying constraints.
                zpd_idx = int(snap_idx)
                break
            if zpd_idx is None:
                zpd_idx = closest_idx
            olap_info = {'avg_class_prob': avg_class_prob,}

        elif match_method == "train_net":
            # ---------------------------------------------------------------- #
            # Re-assign the networks in the ensemble to have a new final layer.
            # Then we re-train that last layer, and compute pair-wise overlaps
            # like we did earlier. We form the training fold splits here for
            # training. I don't think it's a big deal to retrain from scratch
            # (i.e., we don't need to save Adam statistics from training?)
            # ---------------------------------------------------------------- #
            try:
                from distances import utils_dataloaders as UD
                from distances import overlap_dataloader as OD
                from distances import overlap as OP
            except ImportError as e:
                self.logger.info('Cannot import `distances`: {}'.format(e))
            o_acc_all_folds = []
            o_min_all_folds = []

            # ---------------------------------------------------------------- #
            # We need to to save learner's current data into the same directory
            # (pt_data_dir) and same format as teacher's data.  Unfortunately
            # this is going to be tricky with episode splitting.  Note: the
            # student episode df only stores *completed* episodes. So we'll
            # take the last life idx in the df, and anything after is a unique
            # epiosde.  Use `list_of_eps` as list of episodes, to split into
            # *folds* later.  Iterate through episodes and get lives from that.
            # ---------------------------------------------------------------- #
            student_episodes = join(student_logs['dir_episodes'],
                                    'training_summary_true_v02.txt')
            print('calling read_json on: ', student_episodes)
            time.sleep(10)  # no clue if this will work
            assert os.path.exists(student_episodes)
            try:
                eps_df = pd.read_json(student_episodes, orient='index')
            except ValueError as e:
                logging.exception(e)
            eps_df.index = eps_df.index.map(int)
            eps_df.sort_index(inplace=True)
            nb_stored_eps = eps_df.shape[0]
            self.logger.debug('length life buffer: {}, vs ep stored in df: {} '.format(
                    len(learner_buf), nb_stored_eps))
            list_of_eps = []
            state_count = 0  # number of data points in the learner's class.

            # ---------------------------------------------------------------- #
            # Episode in progress. I THINK this should already handle when an
            # episode is completed at the exact time this is called.  Careful,
            # learner_buf is a buffer of LIVES, not episodes! Also, learner_buf
            # keeps increasing in length even if older episodes are evicted.
            # ---------------------------------------------------------------- #
            # NOTE: well OK we only call this after finishing an episode and
            # we'll get a states length of 5 since the episode 'in progress'
            # will have just started. So the code will not execute but just in
            # case we can leave it here if external calls change things.
            # ---------------------------------------------------------------- #
            # Life idx to start, use `nb_stored_eps`, NOT `nb_stored_eps-1`.
            # But annoyingly, eps_df is 1-idxed, but learner_buf isn't (normal
            # list) so we do not apply an extra +1 to get `life_after_stored`.
            # ---------------------------------------------------------------- #

            life_after_stored = eps_df['life_idx_end'][nb_stored_eps]
            self.logger.debug('(0-idx) life_after_stored: {}'.format(life_after_stored))
            current_ep = []
            for l_idx in range(life_after_stored, len(learner_buf)):
                ep_states = learner_buf[l_idx].states
                ep_length = learner_buf[l_idx].length
                self.logger.debug('  l_idx, states, length: {}, {}, {}'.format(
                        l_idx, len(ep_states), ep_length))
                if len(ep_states) <= ignore_k:
                    self.logger.debug('  life {}, length {} so ignore'.format(
                            l_idx, len(ep_states)))
                    continue
                for s_idx in range(len(ep_states)-4, ignore_k, -1):
                    obs = ep_states[s_idx : s_idx + 4]
                    assert obs.shape == (4, 84, 84), obs.shape
                    current_ep.append(obs)
                # We shouldn't hit capacity after the very first episode!
                assert len(current_ep) < num_states, len(current_ep)
            if len(current_ep) > 0:
                state_count += len(current_ep)
                list_of_eps.append(current_ep)

            # ---------------------------------------------------------------- #
            # Handle subsequent episodes, going backwards from last stored.
            # ---------------------------------------------------------------- #
            done = False
            self.logger.debug('Now loading full stored episodes, e_idx is 1-idx wrt df')

            for e_idx in range(nb_stored_eps, 0, -1):
                if done:
                    break
                current_ep = []
                self.logger.debug('e_idx: {}, eps loaded: {}, total c: {}'.format(
                        e_idx, len(list_of_eps), state_count))
                l_start = eps_df['life_idx_begin'][e_idx]
                l_end   = eps_df['life_idx_end'][e_idx]

                # These are 1-idx'd wrt the eps_df, but if we call then on learner_buf,
                # we must annoyingly subtract 1 because learner_buf is a list.
                self.logger.debug('  1-idx life (start,end): {}, {}'.format(l_start,l_end))

                # Eh we'll let all the states in episode e_idx be part of the data.
                # Here we subtract 1 unlike the case when we load from pickle files.
                for l_idx in range(l_start-1, l_end):
                    ep_states = learner_buf[l_idx].states
                    ep_length = learner_buf[l_idx].length
                    if len(ep_states) <= ignore_k:
                        self.logger.debug('  life {}, length {} so ignore'.format(
                                l_idx, len(ep_states)))
                        continue
                    for s_idx in range(len(ep_states)-4, ignore_k, -1):
                        obs = ep_states[s_idx : s_idx + 4]
                        assert obs.shape == (4, 84, 84), obs.shape
                        current_ep.append(obs)
                        state_count += 1
                        if state_count >= num_states and not done:
                            self.logger.debug("  setting done=True, state_count: {}".format(
                                    state_count))
                            done = True
                    self.logger.debug("  done life {} (0-idx) from ep {} (1-idx wrt df)".format(
                            l_idx, e_idx))
                    self.logger.debug("  states/length: {}/{}, state_count {}".format(
                            len(ep_states), ep_length, state_count))
                list_of_eps.append(current_ep)

            # Split into folds.
            epis_indices = np.random.permutation( len(list_of_eps) )
            f_indices = np.array_split(epis_indices, indices_or_sections=num_folds)
            self.logger.debug('Done loading learner data, {} states'.format(state_count))
            self.logger.debug("Splitting list_of_eps (len {}) into {} folds: {}".format(
                    len(list_of_eps), num_folds, f_indices))

            # ---------------------------------------------------------------- #
            # Add data to existing `distances_pytorch_data/fold-k/` directories.
            #   Format:  'snapshot_0XYZ-class-ABC-idx.p'
            #   Example: 'snapshot_0116-class-022-507352.p'
            # The number after class label (ABC) is not used. See how the
            # custom dataloader class obtains the class label. Class:
            # num_models, NOT +1 because num_models is 1-indexed. :-)
            # ---------------------------------------------------------------- #
            # Use XXXX as the key for our new data, which we then delete later.
            # TODO / HUGE NOTE: This will not be safe for multiple teacher
            # student runs w/same teacher, but I don't do that. Actually, never
            # do that, because the data loader would get confused ...
            # ---------------------------------------------------------------- #
            def _syntax_ok(self, x):
                return ('snapshot' in x) and ('class in x') and (x[-2:] == '.p')

            l_items_per_fold = {}

            for f in range(num_folds):
                foldpth = join(pt_data_dir, 'fold-{}'.format(f))
                assert os.path.exists(foldpth), foldpth
                self.logger.debug('Saving at: {}'.format(foldpth))
                i = 0
                for e_idx in f_indices[f]:
                    self.logger.debug('  on episode idx {}'.format(e_idx))
                    episode = list_of_eps[e_idx]
                    for obs in episode:
                        basename = 'snapshot_XXXX-class-{}-{}.p'.format(
                                str(num_models).zfill(3), str(i).zfill(6))
                        data_path = join(foldpth, basename)
                        with open(data_path, 'wb') as fh:
                            pickle.dump(obs, fh)
                        i += 1
                self.logger.debug('  saved {} items in this fold'.format(i))
                l_items_per_fold[f] = i

            # ---------------------------------------------------------------- #
            # Load and build new networks, and create DataLoader.
            # ---------------------------------------------------------------- #
            def _load(pt_path, name, fold):
                pth = join(pt_path, name) + '_fold_{}.p'.format(fold)
                with open(pth, 'rb') as fh:
                    return pickle.load(fh)

            assert num_folds == len(overlap_ensemble)
            new_olap_nets = []
            avg_time = []
            self.logger.debug("l_items_per_fold: {}".format(l_items_per_fold))

            for test_idx,onet in enumerate(overlap_ensemble):
                # Don't forget to load these for overlap computation.
                items_per_class_t = _load(pt_data_dir, 'data-items_per_class_train', test_idx)
                items_per_class_v = _load(pt_data_dir, 'data-items_per_class_valid', test_idx)
                y_valid = np.array(_load(pt_data_dir, 'data-y_valid', test_idx))
                assert y_valid[-1] == y_valid[-2] == y_valid[-3] == (num_models-1), num_models

                # Now we're going to modify the above with overlap statistics.
                # First two just have integer (class index) as keys.
                items_per_class_t[num_models] = sum(
                        [l_items_per_fold[x] for x in l_items_per_fold if x != test_idx]
                )
                items_per_class_v[num_models] = l_items_per_fold[test_idx]
                y_learner = np.ones(l_items_per_fold[test_idx]) * num_models
                y_valid = np.concatenate((y_valid, y_learner))

                # Now the actual DataLoaders.
                self.logger.debug('*** BUILD/TRAIN OVERLAP NET, fold {} ***'.format(test_idx))
                o_data_t = OD.OverlapDataset(pt_data_dir=pt_data_dir,
                                             pt_transform=OD.TRANSFORM_TRAIN,
                                             fold_idx=test_idx,
                                             train=True)
                o_data_v = OD.OverlapDataset(pt_data_dir=pt_data_dir,
                                             pt_transform=OD.TRANSFORM_VALID,
                                             fold_idx=test_idx,
                                             train=False)
                dataloaders = {
                    'train': DataLoader(o_data_t, batch_size=batch_size,
                                        shuffle=True, num_workers=nb_parallel_workers),
                    'valid': DataLoader(o_data_v, batch_size=batch_size,
                                        shuffle=False, num_workers=nb_parallel_workers),
                }
                # Use floats due to dividing by these later, for per-epoch results.
                data_sizes = {'train': float(len(o_data_t)), 'valid': float(len(o_data_v))}

                # Sanity checks. NOTE: these should trigger under high probability if
                # the program crashed before it deleted the XXXX learner files. Good!
                self.logger.debug('  items_per_class_t: {}'.format(items_per_class_t))
                self.logger.debug('  items_per_class_v: {}'.format(items_per_class_v))
                self.logger.debug('  train size: {}'.format(len(o_data_t)))
                self.logger.debug('  valid size: {}'.format(len(o_data_v)))
                self.logger.debug('  sum of two: {}'.format(len(o_data_t)+len(o_data_v)))
                _sums_t = sum([items_per_class_t[x] for x in items_per_class_t])
                _sums_v = sum([items_per_class_v[x] for x in items_per_class_v])
                assert len(o_data_t) == _sums_t, "{} {}".format(len(o_data_t), _sums_t)
                assert len(o_data_v) == _sums_v, "{} {}".format(len(o_data_v), _sums_v)
                assert len(o_data_v) == len(y_valid), \
                        "{} {}".format(len(o_data_v), y_valid.shape)
                assert y_valid[0] == y_valid[1] == y_valid[2] == 0
                assert y_valid[-1] == y_valid[-2] == y_valid[-3] == num_models, num_models

                # For each network, we re-assign so that it has a new architecture.
                # Don't forget that requires_grad=True by default, so set to False.
                new_onet = copy.deepcopy(onet)
                for param in new_onet.parameters():
                    param.requires_grad = False
                new_onet.fc6 = nn.Linear(onet.fc6.in_features, num_models+1)
                new_onet.cuda()

                # This will correctly ignore all params w/`requires_grad=False`.
                # We don't use dropout, but we use the same L2 reg as earlier.
                opt = optim.Adam(new_onet.parameters(), lr=lrate, weight_decay=l2reg)
                criterion = nn.CrossEntropyLoss()

                best_valid_epoch = 0
                best_valid_acc = 0.0
                best_valid_loss = np.float('inf')
                t_start = time.time()

                for e in range(nb_epochs):
                    self.logger.debug('  epoch {}'.format(e))
                    ep_tr_loss,  ep_tr_logits,  ep_tr_pred,  ep_tr_acc  = [], [], [], []
                    ep_val_loss, ep_val_logits, ep_val_pred, ep_val_acc = [], [], [], []

                    b = 0
                    for phase in ['train', 'valid']:
                        if phase == 'train':
                            new_onet.train()
                        else:
                            new_onet.eval()


                        for bb in dataloaders[phase]:
                            b += 1
                            if b % 400 == 0:
                                self.logger.debug('    mb {}'.format(str(b).zfill(5)))
                            inputs = cuda(bb['data'], gpu_params['enabled'],
                                          gpu_params['id'], gpu_params['async'])
                            labels = cuda(bb['label'], gpu_params['enabled'],
                                          gpu_params['id'], gpu_params['async'])
                            opt.zero_grad()
                            with torch.set_grad_enabled(phase == 'train'):
                                outputs_BC = new_onet(inputs)
                                _, preds_B = torch.max(outputs_BC, dim=1)
                                loss = criterion(outputs_BC, labels)
                                if phase == 'train':
                                    loss.backward()
                                    opt.step()

                            # Gather statistics for evaluation later.
                            acc_B = (preds_B == labels)
                            if phase == 'train':
                                ep_tr_loss.append(loss.item())
                                ep_tr_logits.append(outputs_BC.detach().cpu().numpy())
                                ep_tr_pred.extend(preds_B.cpu().numpy())
                                ep_tr_acc.extend(acc_B.cpu().numpy())
                            else:
                                ep_val_loss.append(loss.item())
                                ep_val_logits.append(outputs_BC.detach().cpu().numpy())
                                ep_val_pred.extend(preds_B.cpu().numpy())
                                ep_val_acc.extend(acc_B.cpu().numpy())

                    # Finished one epoch of train/valid so debug/save.
                    ep_tr_loss    = np.mean(ep_tr_loss)           # scalar
                    ep_tr_pred    = np.array(ep_tr_pred)          # vector (#train,)
                    ep_tr_logits  = np.concatenate(ep_tr_logits)  # matrix (#train,classes)
                    ep_val_loss   = np.mean(ep_val_loss)          # scalar
                    ep_val_pred   = np.array(ep_val_pred)         # vector (#valid,)
                    ep_val_logits = np.concatenate(ep_val_logits) # matrix (#valid,classes)
                    L0 = len(ep_tr_acc)
                    L1 = len(ep_val_acc)
                    assert L0 == len(ep_tr_pred), "{} vs {}".format(L0, len(ep_tr_pred))
                    assert L1 == len(ep_val_pred), "{} vs {}".format(L1, len(ep_val_pred))
                    assert ep_tr_logits.shape == (L0, num_models+1), ep_tr_logits.shape
                    assert ep_val_logits.shape == (L1, num_models+1), ep_val_logits.shape
                    self.logger.debug("  done epoch: {}".format(e))
                    self.logger.debug("train loss: {:.5f}".format(ep_tr_loss))
                    self.logger.debug("valid loss: {:.5f}".format(ep_val_loss))
                    self.logger.debug("train acc:  {:.1f}  ({} / {})".format(
                            100.*np.sum(ep_tr_acc)/L0, np.sum(ep_tr_acc), L0))
                    self.logger.debug("valid acc:  {:.1f}  ({} / {})".format(
                            100.*np.sum(ep_val_acc)/L1, np.sum(ep_val_acc), L1))
                    self.logger.debug("vpreds:     ({:.2f}, {:.2f}), {:.3f} +/- {:.1f}".format(
                            np.min(ep_val_pred), np.max(ep_val_pred),
                            np.mean(ep_val_pred), np.std(ep_val_pred))
                    )
                    this_val_acc = np.sum(ep_val_acc)/L1

                    # If best valid, store these predictions, which we use in overlaps.
                    if best_valid_acc < this_val_acc:
                        self.logger.debug("best_valid ({:.5f}, epoch {}) < this_val_acc: "
                            "{:.5f})".format(best_valid_acc, best_valid_epoch, this_val_acc))
                        best_valid_acc     = this_val_acc
                        best_valid_loss    = ep_val_loss
                        best_valid_epoch   = e
                        # We can do this here but for now let's just do 2 epochs each time.
                        #ep_val_logits_best = ep_val_logits
                    else:
                        pass
                    current_epoch_valid_logits = ep_val_logits # Use for overlaps.
                    # Back to the next epoch. :-)

                t_time = (time.time() - t_start) / 60.
                self.logger.debug('o-lap fold train/valid time: {:.1f} mins'.format(t_time))
                avg_time.append(t_time)

                # Set to evaluation mode and save, but I don't think this matters
                # much as we already have the validation predictions from earlier.
                new_onet.eval()
                new_olap_nets.append(new_onet)


                # ---------------------------------------------------------------- #
                # OVERLAP OVERLAP! See `plot/dqn_multiclass_plot_ratios_with_folds.py`.
                # We want the *multiclass* overlap, not the binary one. The
                # num_models is the number of classes, which we increment by 1. The
                # `this_model` should be the learner's model as we want overlap wrt
                # the learner, for all the OTHER snapshots (from the teacher).
                # Need keys for valid labels, num per class, and valid LOGITS.
                # Normally we'd use the best but I want to give at least 2 epochs
                # each time we do this ... so use `current_epoch_valid_logits`.
                # ---------------------------------------------------------------- #
                f_data = {}
                f_data['ep_val_labels'] = y_valid
                f_data['num_per_class_valid'] = items_per_class_v
                f_data['ep_val_logits_best'] = current_epoch_valid_logits
                oo = OP.overlap_multi(data=f_data,
                                      num_models=num_models+1, # total classes
                                      this_model=num_models)   # assume last class idx
                o_acc_all_folds.append( oo['acc_vector'] )
                o_min_all_folds.append( oo['min_vector'] )
                self.logger.debug('(debugging) o_acc: {}'.format(np.array(oo['acc_vector'])))
                self.logger.debug('o_min: {}'.format(np.array(oo['min_vector'])))
                # Whew, finished w/stuff for this fold. Go to next fold...

            # Done with all folds. Report timing:
            self.logger.debug('with {} epochs, {} batchsize, {} workers ... '
                'average o-lap train/valid time: {:.2f} +/- {:.1f} mins'.format(
                    nb_epochs, batch_size, nb_parallel_workers,
                    np.mean(avg_time), np.std(avg_time)))

            # Shape of o_acc, o_min will be (num_folds, num_classes)
            o_acc = np.array(o_acc_all_folds)
            o_min = np.array(o_min_all_folds)
            assert o_acc.shape == o_min.shape
            assert o_acc.shape == (num_folds, num_models+1)
            o_acc_mean = np.mean(o_acc, axis=0)
            o_acc_std = np.std(o_acc, axis=0)
            o_min_mean = np.mean(o_min, axis=0)
            o_min_std = np.std(o_min, axis=0)
            self.logger.debug('avg olap acc: {}'.format(o_acc_mean))
            self.logger.debug('     acc std: {}'.format(o_acc_std))
            self.logger.debug('avg olap min: {}'.format(o_min_mean))
            self.logger.debug('     min std: {}'.format(o_min_std))

            # ---------------------------------------------------------------- #
            # Finally, determine the next ZPD snapshot given these overlap metrics.
            # Note: differences *includes* the last model idx, i.e. *LEARNER* model,
            # but the idx_candidates will not include it, so the `zip` ignores it.
            # Code is similar to that of the probability targets case earlier.
            # ---------------------------------------------------------------- #
            self.logger.debug('Now determining ZPD snapshot! Current matched '
                'idx: {}. Overlap targ: {}'.format(closest_idx, overlap_target))
            differences = np.abs(overlap_target - o_min_mean)  # MIN!!
            idx_candidates = condense_f * (1 + np.arange(num_models))
            diff_idx = [(x,y) for (x,y) in zip(differences,idx_candidates)]
            diff_idx = sorted(diff_idx, key=lambda x:x[0])
            self.logger.debug('differences (len {}): {}'.format(len(differences),differences))
            self.logger.debug('diff_idx (len {}): {}'.format(len(diff_idx),diff_idx))

            # Create zpd_idx in case the loop might not assign to it.
            zpd_idx = None
            for dd,snap_idx in diff_idx:
                # Ignore snapshots with lower index than current matched one.
                if snap_idx <= closest_idx:
                    continue
                # Ignore snapshots with lower reward than current value.
                if diff[snap_idx] < 0:
                    continue
                # Otherwise we're at the closest snapshot idx satisfying constraints.
                zpd_idx = int(snap_idx)
                break
            # https://github.com/CannyLab/dqn/issues/65
            if zpd_idx is None:
                #zpd_idx = closest_idx
                self.logger.debug('zpd_idx None, so setting to last of idx_candidates')
                zpd_idx = int(idx_candidates[-1])
            self.logger.debug('setting zpd_idx: {}'.format(zpd_idx))

            # For reporting later.
            olap_info = {'avg_olap_acc_mean': o_acc_mean,
                         'avg_olap_acc_std': o_acc_std,
                         'avg_olap_min_mean': o_min_mean,
                         'avg_olap_min_std': o_min_std}

            # ---------------------------------------------------------------- #
            # FINALLY, delete data files we just created from learner's states.
            # BE CAREFUL, the 'XXXX' is how we designate a learner obs/state.
            # ---------------------------------------------------------------- #
            for f in range(num_folds):
                foldpth = join(pt_data_dir, 'fold-{}'.format(f))
                assert os.path.exists(foldpth), foldpth
                f_obs = [join(foldpth,x) for x in os.listdir(foldpth) if 'XXXX' in x]
                self.logger.debug('Removing {} files from: {}'.format(len(f_obs), foldpth))
                for ff in f_obs:
                    os.remove(ff)
            self.logger.debug('Whew, done with overlap shenanigans ...')
        else:
            raise ValueError(match_method)

        self.logger.debug("last_match_idx: {}  (zero-indexed)".format(self.last_match_idx))
        self.logger.debug("learner reward: {:.2f}".format(value))
        self.last_match_idx = closest_idx
        self.last_zpd_idx = min(zpd_idx, len(self) - 1)
        self._has_matched_to_something = True
        return self.last_match_idx, self.last_zpd_idx, olap_info

    def match_with_rewards(self, rewards: float) -> int:
        """Find first snapshots w/rewards at least as big as `rewards`.

        From SnapshotDistillation, not the MultiTeacherTeachingCenter.
        From Allen's old code, we should fix if we want to use.
        """
        diff = self._data_frame - rewards
        try:
            match_idx = diff[diff > 0].index[0]
        except IndexError:
            raise StopIteration("Can't match any more values in the summary.")
        self.last_match_idx = match_idx
        self.last_zpd_idx = min(
            self.condensed_last_idx, match_idx + self._lead_number)
        return self.last_match_idx
