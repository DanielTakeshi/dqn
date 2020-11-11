import copy
import torch.nn as nn
import dqn
from dqn.utils.variables import create_var, create_batch
from dqn.utils.train import merge_transitions
from dqn.agent.agent import Agent
import numpy as np
import torch
from dqn.utils.io import write_dict
from dqn.utils.data_structures import SMAQueue
from dqn.teacher.snapshots_teacher import SelfReviewer
from dqn.utils.math import sigmoid
from dqn.replay.transition import Transition


class DQNAgent(Agent):
    """Subclasses agent.Agent.

    Supeclass provides `save_model` and `get_policy_epsilon` methods, and has
    references to the logger, network, and policy.

    DQNTrainAgent: used for atari.py when we call `agent.train(steps)`.
    DQNAgent: used for train/test processes, since they should not need the
        training stuff, but still need the _calling_ mechanism. Well,
        technically they pass the DQNAgent to the ExperienceSource, and _that_
        does the env stepping.
    """
    def __init__(self, net, gpu_params, log_params, policy=None, tag=""):
        super(DQNAgent, self).__init__(
            net=net, gpu_params=gpu_params, log_params=log_params,
            policy=policy, tag=tag)
        self.logger.debug("A playing DQN agent has been created.")

    def _states_preprocessor(self, states):
        assert states.shape[0] == 1
        return create_var(states, gpu=self._gpu_params["enabled"],
                          gpu_id=self._gpu_params["id"],
                          requires_grad=False,
                          gpu_async=self._gpu_params["async"],
                          convert_func="numpy")

    @staticmethod
    def _qs_postprocessor(qs):
        assert qs.shape[0] == 1
        return qs.detach().cpu().numpy()

    def __call__(self, states, steps=None):
        """
        Given the states as the input, it will run through an epsilon-greedy
        policy based on the Q values returned from the neural network.

        :param states: A `numpy` array of the states input
        :param steps: Current total steps to control epsilon-greedy
        :return: action picked by policy
        """
        obs = self._states_preprocessor(np.expand_dims(states, 0))
        return self._policy(self._qs_postprocessor(self._net(obs)), steps=steps)


class DQNTrainAgent(DQNAgent):
    """What we actually use for training DQN normally.

    Provides bellman loss and stuff more specific to DQN.  Also provides some
    DQfD-like support, e.g., wrt the large-margin supervised loss.
    """

    def __init__(self, net, gpu_params, log_params, opt, train_params, replay,
                 policy=None, teacher=None, avg_window=100, tag=""):
        super(DQNTrainAgent, self).__init__(
            net=net, gpu_params=gpu_params, log_params=log_params,
            policy=policy, tag=tag)
        self._target = copy.deepcopy(net)
        self._avg_window = avg_window
        self._opt = opt
        self._train_params = train_params
        self._replay = replay
        self._loss_func = nn.MSELoss(reduce=False)
        self._teacher = teacher
        self._debug_loss = {}
        self._train_step_counts = 0
        self._teacher_samples = {}
        self.self_reviewer = None
        self._timer = {}
        self._steps_after_done = 0
        self.logger.debug("A training DQN agent has been created.")

    def get_current_lr(self, steps):
        """Use for logging to report learning rate.

        See example usage in `experiments/atari.py`.
        """
        return self._opt.get_lr(steps)

    def get_teacher_blend(self, steps):
        """Use for logging to report teacher blend.

        See example usage in `experiments/atari.py`. If we are done matching,
        then we actually want to use a different blend schedule.
        """
        # TODO: change if condition, see comments in `sample_transitions()`.
        if self._teacher.any_teacher_done():
            return self._teacher.blend_schedule_end.value(self._steps_after_done)
        else:
            return self._teacher.blend_schedule.value(steps)

    def get_wis_beta(self, steps):
        """Use for logging to report the importance sampling beta term.

        See example usage in `experiments/atari.py`. Assumes we can extract
        teacher via the dict `_teachers`.
        """
        return (self._teacher)._teachers[0].beta_schedule.value(steps)

    @property
    def debug_loss(self):
        return self._debug_loss

    def sync_target(self, steps):
        """ Synchronize target network.
        :param steps: Current total steps
        """
        self._target.init_weight(self._net.state_dict())
        self.logger.debug("Target network has been synchronized at steps "
                          "{0}.".format(steps))

    def sample_transitions(self, steps):
        """Sample transitions from replay buffer.

        Supports normal training only (no teacher, first case) vs student and
        teacher training. For the latter it allocates a fraction of the batch
        size from the `blend_schedule` schedule, but eventually we should just
        use prioritization for this. Note that to detect if we have a teacher at
        all, there are two ways of doing so.

        Recall that `self._teacher` is a `MultiTeacherTeachingCenter`, at least
        with single student-teacher training.

        Returns
        -------
        (samples, learner_bs): (`Transition` w/minibatch of samples, int)
        - `Transition` object, with learner samples followed by teacher samples.
        - int representing number of minibatches samples from learner's history.
        """
        bs = self._train_params["batch_size"]
        ns = self._train_params["num_steps"]

        if self._teacher is None or self._teacher.total_num_teachers == 0:
            samples = self._replay.sample(batch_size=bs, num_steps=ns)
            return samples, bs
        else:
            # AH! Let's insert a special case if we've finished matching.
            # TODO: actually won't work if we have multiple teachers, but just
            # change the if condition below if we end up doing that.
            if self._teacher.any_teacher_done():
                tb = self._teacher.blend_schedule_end.value(self._steps_after_done)
                self._steps_after_done += 1
            else:
                tb = self._teacher.blend_schedule.value(steps)
            teacher_bs = int(bs * tb)
            learner_bs = bs - teacher_bs

            # Sample from the teacher's replay.
            teacher_samples, teacher_id = self._teacher.sample(
                    batch_size=teacher_bs, num_steps=ns, steps=steps)
            if teacher_samples is None:
                teacher_bs = 0
                learner_bs = bs
            else:
                assert isinstance(teacher_samples, Transition)
                assert len(teacher_samples.action) == teacher_bs
                # Used to be here but DON'T CALL IT! An absurd amount of compute.
                #self._add_teacher_samples(teacher_id, teacher_samples)

            # Sample from the learner's replay (same class, different object).
            learner_samples = self._replay.sample(batch_size=learner_bs, num_steps=ns)
            assert isinstance(learner_samples, Transition)
            assert len(learner_samples.action) == learner_bs
            self._replay.writer.add_scalar(
                    tag="teacher_sample_percentage",
                    scalar_value=teacher_bs/bs,
                    global_step=steps)
            samples = merge_transitions(learner=learner_samples,
                                        teacher=teacher_samples)
            return samples, learner_bs

    def get_weights(self, states, actions, weights):
        """D: with my pong snapshots code, we're in the second if case.
        Thus, `weights` just gets put into a torch variable, representing the
        importance sampling correction weights.
        """
        if self._teacher and self._teacher.is_weight_teacher:
            return self._teacher.weight(states, actions)
        else:
            return create_var(weights,
                              gpu=self._gpu_params["enabled"],
                              gpu_id=self._gpu_params["id"],
                              gpu_async=self._gpu_params["async"],
                              convert_func="numpy").float()

    def _add_new_loss(self, value, label):
        """Use for debugging to record loss values.

        In `atari.py`, we iterate through different losses here and report them.
        Save in `info_summary.txt`. These are queues which keep a value in a
        past window (usually around 100); when we log we usually take the mean.

        Losses can be somewhat misleading; after a student finishes matching a
        teacher, there are no more supervised losses, yet we will keep returning
        the older values since we don't update the queue. But it's fine for now.
        """
        if label not in self._debug_loss:
            self._debug_loss[label] = SMAQueue(size=self._avg_window)
        self._debug_loss[label] += value.mean().item()

    def loss(self, steps):
        """Computes loss vector for optimization later in `self.opt_step()`.

        (1) Samples from buffer, but w/knowing which data was from learner vs
        teacher; if no teacher, `learner_bs=None`. In teacher-student
        minibatches, the student's samples (if any) come first, THEN the
        teacher's samples (if any) are second.

        (2) Create minibatch with 1-step returns on GPU, though we might do only
        n-step later; we don't currently have capability to support mixing the
        two. The `create_batch` argument `requires_grad` only affects current
        state, `transitions.state`, when we create it via `Variable(...)`. Here
        it's false since the states are not 'trainable' -- but for progress
        score computation, we *do* set `requires_grad=True`.

        (3) Compute Bellman loss, requires a forward pass through net. Applied
        for BOTH DQN-like and DQfD-like agents.

        (4) Optionally, add the teacher large-margin loss, i.e., the supervised
        loss, using the teacher sample portion of the minibatch ONLY. Well,
        technically, self-generated data has lambda=0 so the loss is 0, and we
        track that for all items in the minibatch. The supervise margin was 0.8
        in DQfD.  Compare with `AtariProgressNet_progress_backward` in
        `models.py`, which also calls the same loss function. (If using progress
        scores, we need the progress net to implement the same loss function.)

        (5) Note that DQfD used L2 regularization, with lambda as 10^{-5}, or
        1e-5 in Python.

        Returns
        -------
        A tuple, (loss,weights): both are of shape (B,), NOT SCALARS! First
            term for loss (potentially including imitation/supervised loss) and
            second for importance sampling corrections.
        """
        # (1) Sample from buffer.
        transitions, learner_bs = self.sample_transitions(steps=steps)
        apply_extra_losses = (learner_bs != self._train_params['batch_size'])

        # (2) Create minibatch via PyTorch tensors.
        states, next_states, actions, rewards, dones = create_batch(
                transitions=transitions,
                gpu=self._gpu_params["enabled"],
                gpu_id=self._gpu_params["id"],
                gpu_async=self._gpu_params["async"],
                requires_grad=False)
        weights = self.get_weights(states, actions, transitions.weight)

        # (3) Bellman loss.
        loss, qs, qs_full = dqn.utils.train.bellman_loss(
                net=self._net,
                target_net=self._target,
                loss=self._loss_func,
                states=states,
                next_states=next_states,
                actions=actions,
                rewards=rewards,
                dones=dones,
                num_steps=self._train_params['num_steps'],
                double_dqn=self._train_params["double"],
                gamma=self._train_params["gamma"])
        self._add_new_loss(loss, "bellman_loss")
        self._add_new_loss(loss[:learner_bs], "bellman_from_student")
        if apply_extra_losses:
            self._add_new_loss(loss[learner_bs:], "bellman_from_teacher")

        # (4) Teacher large-margin (i.e., supervised) loss, including lambda.
        if self._teacher and self._teacher.supervise_loss and apply_extra_losses:
            _supervise_loss = dqn.utils.train.supervise_loss(
                    qs=qs,
                    qs_full=qs_full,
                    actions=actions,
                    margin=self._teacher.supervise_margin,
                    gpu=self._gpu_params["enabled"],
                    gpu_id=self._gpu_params["id"],
                    gpu_async=self._gpu_params["async"],
                    loss_lambda=self._teacher.supervise_loss_lambda,
                    loss_type=self._teacher.supervise_type,
                    learner_bs=learner_bs)
            self._add_new_loss(_supervise_loss[learner_bs:], "supervise_loss")
            loss += _supervise_loss

        return loss, weights

    def opt_step(self, steps, loss, weights):
        """I think this is the way to do DQN clipping/stepping in PyTorch.

        Could probably double check one of these days but a low priority.

        The backward pass requries an extra tensor, `weights`, since `loss` is
        of size (B,). From PyTorch docs:

        > If you want to compute the derivatives, you can call .backward() on a
        > Tensor. If Tensor is a scalar (i.e. it holds a one element data), you
        > don’t need to specify any arguments to backward(), however if it has
        > more elements, you need to specify a gradient argument that is a
        > tensor of matching shape.

        Here, `weights` conveniently acts as our importance sampling correction,
        which is folded in the normal Q-Learning loss (+teacher loss if needed).
        Even though these are vectors, the optimizer _should_ correctly apply
        all of these MB elements and adjust the parameters based on all samples.
        """
        opt = self._opt.get_opt(steps)
        opt.zero_grad()
        loss.backward(weights)
        if self._opt.clipping != 0:
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), self._opt.clipping)
        opt.step()

    def train(self, steps):
        """Called from `atari.py` script, main training method.

        Computes loss followed by optimization step, and then if necessary, sync
        with target network, quite frequently. It's wrt `steps`, and this is
        indeed the same as an environment step that I've been using in my other
        code, assuming that the `play` process called during `atari.py` is a
        single env and not a set of parallel envs.

        This is only called every 4 steps; train is called w/steps at 10004,
        10008, 10012, etc., assuming the first 10k is just for populating the
        replay buffer. BUT, if condition depends on `self._train_step_counts`,
        which increments once for each 4 times steps is incremented. Thus, if we
        set target sync at N, then we are actually (if this were DeepMind code)
        syncing at 4N steps. Not too big of a deal as long as it's roughly
        10k-40k ish. In Double DQN paper, they say they synced target network
        at 10k steps.
        """
        loss, weights = self.loss(steps)
        self.opt_step(steps, loss, weights)
        self._train_step_counts += 1
        if self._train_step_counts % self._train_params["target_sync_per_step"] == 0:
            self.update_teacher_weights()
            self.sync_target(steps)
            self._timer[steps] = self._teacher.timer()
            self._teacher.reset_timer()
            write_dict(dict_object=self._timer,
                       dir_path=self._log_params["dir"],
                       file_name="download_summary")
            self._train_step_counts = 0

    def update_teacher_weights(self):
        """Update teacher weights if we have more than one teacher.

        Not applied for normal teacher-less training runs, or for one teacher
        case. (And if there are no samples?)
        """
        if self._teacher is None or self._teacher.total_num_teachers <= 1 or \
                len(self._teacher_samples) == 0:
            return

        #TODO: check, can we remove this? It relies on `self._teacher_samples`
        # but my profiling showed this was taking up an extraordinary amount of
        # computational time becasue we keep stacking ALL transitions from the
        # teacher's samples!

        if self.self_reviewer is None:
            self.self_reviewer = SelfReviewer(
                obs_space=self._net.obs_space,
                num_actions=self._net.num_actions,
                train_params=self._train_params,
                gpu_params=self._gpu_params,
                teacher_params=self._teacher.teacher_params,
                total_teachers=self._teacher.total_num_teachers,
                opt_params=self._opt.params,
                max_num_steps=self._policy.max_num_steps)
        _scores, _num_transitions = self.self_reviewer.examine(
            net=self.get_net(), target=self._target,
            transitions=self._teacher_samples)
        # _scores = (1 - np.sign(_scores))/2
        _scores = sigmoid(_scores)
        _update_weight = np.power(1 - self._teacher.eta, _scores)
        _old_weight = self._teacher.teacher_weights
        self._teacher.teacher_weights = np.multiply(_old_weight, _update_weight)
        self.logger.info(
            "Multiplicative weights updates: \nNumber of transitions: {0}\n"
            "Old weights: {1} \nNew weights: {2} \nCosts (0-1): {3}".format(
                _num_transitions, _old_weight, self._teacher.teacher_weights,_scores))
        self._teacher_samples = {}
