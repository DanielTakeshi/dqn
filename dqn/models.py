import torch.nn as nn
from dqn.utils.variables import create_var
import math
import torch.nn.functional as func
import torch
import numpy as np
from dqn.utils.variables import create_batch
import dqn.utils.train
from dqn.utils.math import softmax_deepmind as convert_to_dist
from dqn.optimizer import Optimizer


class AtariNet(nn.Module):
    def __init__(self, obs_space, num_actions, hidden_size=512):
        """
        Original Deepmind architecture for Atari games.

        :param obs_space: Frame size from Gym environment, usually a tuple
        (A, B, C) where A is the number of stacked frames, B and C are the
        dimensions of the images.
        :param num_actions: Number of actions available in the environment
        :param hidden_size: Number of hidden units in the last fully
        connected layer. It is actually different across a few papers so that
        here becomes an parameter.
        """
        super(AtariNet, self).__init__()
        assert isinstance(obs_space, tuple)
        assert isinstance(num_actions, int)
        assert num_actions > 0
        assert isinstance(hidden_size, int)
        assert hidden_size > 0
        self.num_actions = num_actions
        self.obs_space = obs_space
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(obs_space[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(64 * 7 * 7, self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size, self.num_actions)

    def init_weight(self, weights=None):
        """
        Initialize the weights of the neural network either through loading
        an existing model parameters or Xavier normal initialization.

        :param weights: A dict containing parameters and persistent buffers.
        If not loading from existing model, it should be set to `None`.
        """
        if weights is not None:
            self.load_state_dict(weights)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def forward(self, x, activation=False):
        """
        Inherit forward function to compute the neural network output given
        the input `x`. This function also has an option to output the
        activation layer, which is the input to the last fully connected layer.

        D: in normal DQN, we just call `net(states)` so `activation=False`.

        :param x: Input tensor, the first dimension is the batch size where
        the remaining should be the same as `obs_space`.
        :param activation: boolean variable to denote if output the
        activation from the input tensor.
        :return: If `activation` is set to True, it will output both the
        output from the neural network and activations in a tuple. If False,
        it will simply only output the neural network outputs.
        """
        _batch_size = x.size(0)
        x = x.float() / 255.0
        x = func.relu(self.conv1(x))
        x = func.relu(self.conv2(x))
        x = func.relu(self.conv3(x))
        x = x.view(_batch_size, -1)
        x = func.relu(self.fc4(x))
        if activation:
            return self.fc5(x), x
        else:
            return self.fc5(x)


class AtariProgressNet(nn.Module):
    """D: model for teacher, with capability for computing progress scores.

    Has upper and lower parts of the network.
    upper: index 0, contains current snapshot weights.
    lower: index 1, contains optimal snapshot weights, loaded from best teacher
        snapshot at the very beginning of training
    """
    def __init__(self, obs_space, num_actions, max_num_steps, train_params,
                 gpu_params, teacher_params, opt_params):
        super(AtariProgressNet, self).__init__()
        self.num_actions = num_actions
        self.obs_space = obs_space
        self._train_params = train_params
        self._gpu_params = gpu_params
        self._teacher_params = teacher_params
        self.conv1 = nn.ModuleList(
            [nn.Conv2d(obs_space[0], 32, kernel_size=8, stride=4),
             nn.Conv2d(obs_space[0], 32, kernel_size=8, stride=4)])
        self.conv2 = nn.ModuleList(
            [nn.Conv2d(32, 64, kernel_size=4, stride=2),
             nn.Conv2d(32, 64, kernel_size=4, stride=2)])
        self.conv3 = nn.ModuleList(
            [nn.Conv2d(64, 64, kernel_size=3, stride=1),
             nn.Conv2d(64, 64, kernel_size=3, stride=1)])
        self.fc4 = nn.ModuleList(
            [nn.Linear(64 * 7 * 7, self._train_params["hidden_size"]),
             nn.Linear(64 * 7 * 7, self._train_params["hidden_size"])])
        self.fc5 = nn.ModuleList(
            [nn.Linear(self._train_params["hidden_size"], num_actions),
             nn.Linear(self._train_params["hidden_size"], num_actions)])
        self._opt = Optimizer(
            net=self,
            opt_params=opt_params, max_num_steps=max_num_steps,
            train_freq_per_step=train_params["train_freq_per_step"])
        self._layers = [self.conv1, self.conv2, self.conv3, self.fc4, self.fc5]
        self._resizing_layer_idx = 3
        self._relu_layer_idx = [0, 1, 2, 3]
        self.lower_nodes = None
        self.upper_nodes = None
        self.upper_grads = None
        self.bias_diff = None
        self.lower_bias = None


    def init_intermediate_variables(self):
        """D: _not_ the weights of the net, but _nodes_ and _gradients_.
        See progress score computation for usage. We don't use `lower_grads`
        because we get those via `lower_grads[i].grad`.
        """
        self.lower_nodes = [None] * 5
        self.upper_nodes = [None] * 5
        self.upper_grads = [None] * 6


    @staticmethod
    def _layer_name(name, upper=True):
        """D: helper for `_load_weight` to get naming for upper vs lower.
        """
        _name = name.copy()
        if upper:
            _name.insert(1, "0")
        else:
            _name.insert(1, "1")
        return ".".join(_name)


    def _load_weight(self, model, upper=True):
        """D: Standard in PyTorch, load by assigning to `state_dict`.
        """
        own_state = self.state_dict()
        for name, param in model.items():
            _name = name.split(".")
            upper_name = self._layer_name(_name, upper=True)
            lower_name = self._layer_name(_name, upper=False)
            _name = upper_name if upper else lower_name
            if _name in own_state:
                own_state[_name].copy_(param)


    def zero_lower_bias(self):
        """D: called at the _start_ of `self.progress_scores()`.

        Zero bias parameters in the lower part. Iterate through all the names,
        so all lower _and_ upper variables. If it's a lower (with index 1 in
        name) then we zero it out by assigning zero to the state_dict. But
        before this, we need to keep a copy in `self.lower_bias` so we can later
        revert, because otherwise we change the meaning of what the policy would
        do ... (the _optimal_ one, that is, since it's the lower part).

        We also store `self.bias_diff`, so for each _layer_, which is half of
        the items in the state dict since we really have two params in one
        layer. Compute (upper_bias - lower_bias) and store it. AH, this is for
        the progress scores since we have (dL/dX - dL/dX_0), and the bias is
        part of those two gradient terms.

        This is necessary to truly pass in '0' as input to the lower part, as
        Allen explains in the paper. If not, we'd get an incorrect output value.
        """
        for name, param in self.state_dict().items():
            _name = name.split(".")
            layer_num = int(_name[0][-1])
            if "bias" in _name and int(_name[1][-1]) == 1:
                _name[1] = "0"
                _upper_name = ".".join(_name)
                self.lower_bias[name] = param.clone()
                self.bias_diff[layer_num] = self.state_dict()[_upper_name] - \
                    self.lower_bias[name]
                param *= 0
                self.state_dict()[name].copy_(param)


    def revert_lower_bias(self):
        """D: called at the _end_ of `self.progress_scores()`.
        Simple, we kept a copy of biases in `lower_bias` and simply re-assign.
        Interestingly, we don't reset `self.bias_diff` here ...
        """
        for name, param in self.lower_bias.items():
            self.state_dict()[name].copy_(param)
        self.lower_bias = {}


    def load_upper_weight(self, model):
        """D: load model for _upper_ part, i.e., matched snapshot.
        Also reset the `bias_diff` since that now changes with new upper stuff.
        """
        self.bias_diff = {}
        self._load_weight(model, upper=True)


    def load_lower_weight(self, model):
        """D: load model for _lower_ part, i.e., optimal snapshot.
        Also reset the `bias_diff` since that now changes with new lower stuff.
        """
        self.bias_diff = {}
        self.lower_bias = {}
        self._load_weight(model, upper=False)


    def sync_lower_weight(self):
        """D: makes lower weights equal to the upper weights, but never called?
        """
        for name, param in self.state_dict().items():
            _name = name.split(".")
            if "bias" in _name and int(_name[1][-1]) == 1:
                _name[0] = "0"
                _upper_name = "".join(_name)
                self.state_dict()[_name].copy_(self.state_dict()[_upper_name])


    def load_weight(self, upper, lower):
        """D: If we want to load both weights. But, watch out for zero bias.
        It's only used in `SelfReviewer` which I think is for analysis purposes.
        """
        self.load_upper_weight(upper)
        self.load_lower_weight(lower)
        self.zero_lower_bias()


    def save_gradient(self, idx):
        """D: Called from `self.forward` for computing progress scores.

        Returns a _function_, so use like:
            `x.register_hook(self.save_gradient(i))`
        Where `idx=i` is the _layer_ index of ProgressNet.

        Stores intermediate gradients in `upper_grads` so we can directly query
        when computing the dL/dX term in (dL/dX - dL/dX_0).
        """
        def hook(grad):
            self.upper_grads[idx] = grad
        return hook


    def half_forward(self, x, activation=False, upper=True):
        """D: runs a forward pass only through _one_ part of the progress net,
        either upper or lower. Otherwise, it's similar to the AtariNet method.
        Normally we do the upper part.

        Parameters
        ----------
        x: Input tensor, which in this case should be samples from a teacher's
            replay buffer.
        activation: As in normal AtariNet, if True, return a tuple consisting
            of the output _and_ the activations, i.e., input to the last layer.
        upper: Whether to use upper part or lower art of progress net.
        """
        if upper:
            idx = 0
        else:
            idx = 1
            assert len(self.lower_bias) == 0
        _batch_size = x.size(0)
        x = x.float() / 255.0
        x = func.relu(self.conv1[idx](x))
        x = func.relu(self.conv2[idx](x))
        x = func.relu(self.conv3[idx](x))
        x = x.view(_batch_size, -1)
        x = func.relu(self.fc4[idx](x))
        if activation:
            return self.fc5[idx](x), x
        else:
            return self.fc5[idx](x)


    def forward(self, x, activation=False):
        """D: called to get Q-values from `_progress_backward`. Uses same
        parameters as `self.half_forward`.

        Iterate through the layers (five by default). For each, we explicitly
        assign 'lower_nodes' to have a _zero_ tensor. Upper nodes get `x`.
        (But this applies to _all_ the layers, and `x` is continually updated
        via the for loop.)

        It's confusing why we pass in zero to the lower nodes, BUT, these
        tensors have `requires_grad=True` so should to track relevant gradients.
        Other advantage: the actual output (return value `x`) is as if we had
        passed _only_ through the upper part, i.e., only the theta_i.

        We sum the lower and upper networks as shown in Allen's paper; only
        reason for the 'if' case is due to flattening as needed.

        'Activation' follows usual meaning in similar code, returns the
        activations right before applying the last FC layer.
        """
        self.init_intermediate_variables()
        x = x.float() / 255.0
        bs = x.shape[0]
        _activation = None
        for i in range(len(self._layers)):
            layer = self._layers[i]
            self.lower_nodes[i] = create_var(
                x=x.shape,
                gpu=self._gpu_params["enabled"],
                gpu_id=self._gpu_params["id"],
                gpu_async=self._gpu_params["async"],
                convert_func="torch_zeros",
                requires_grad=True)
            self.upper_nodes[i] = x
            if i == 0:
                x.register_hook(self.save_gradient(i))
            if i == self._resizing_layer_idx:
                x = layer[0](self.upper_nodes[i].view(bs, -1)) + \
                    layer[1](self.lower_nodes[i].view(bs, -1))
            else:
                x = layer[0](self.upper_nodes[i]) + \
                    layer[1](self.lower_nodes[i])
            x.register_hook(self.save_gradient(i+1))
            if i in self._relu_layer_idx:
                x = func.relu(x)
            if i == len(self._layers) - 2 and activation:
                _activation = x
        if activation:
            return x, _activation
        else:
            return x


    def step(self, transitions, steps):
        """D: alternative call to _progress_backward, but I think only for the
        multi *teacher* case.
        
        From `SnapshotDistillation`'s: `matching` => `_score_teacher_replays`
        then `SnapshotsTeacher's`: `progress_step`.  But only applies if more
        than one XP replay there, so definitely for multiple teachers.
        """
        self._progress_backward(transitions=transitions, naive=True)
        # assert self.conv1[1].grad is None
        self._opt.get_opt(steps).step()


    def _progress_backward(self, transitions, naive=False):
        """D: called from `self.progress_scores` w/a batch of XP transitions;
        note that these are _teacher_-provided samples.

        First, get minibatch of samples as usual, BUT set `requires_grad=True`;
        only affects `transitions.state`, and not the successors and other
        things. It makes the Variable() wrapper around the state 'trainable'.

        Calls `get_qs`, returns q-values for actions chosen (`qs`) along with
        full vector/matrix `qs_full`. Get activations via the _upper_ part, and
        then `target_qs` also from the activations, though here the target net
        is the same, since we assume there's no 'lag' that we need for stability
        (Think about if this makes sense?)
        
        Then compute the normal loss, since we want the gradients for that, done
        with `_loss.backward()`. That's the `L_theta(s)` in Allen's paper.

        Applies `_loss.backward()`, which should supply the variables with
        gradients (via `.grad`) that we use later. Returns the _activations_. 
        """
        states, next_states, actions, rewards, dones = create_batch(
            transitions=transitions, gpu=self._gpu_params["enabled"],
            gpu_id=self._gpu_params["id"], gpu_async=self._gpu_params["async"],
            requires_grad=True)
        qs, qs_full = dqn.utils.train.get_qs_(
            self, states=states, actions=actions, upper_forward=naive)
        activations = dqn.utils.train.get_activation(self, states, upper=True)
        target_qs = dqn.utils.train.get_qs_target(
            target_net=self, net=self, next_states=next_states,
            rewards=rewards, dones=dones, gamma=self._train_params["gamma"],
            num_steps=self._train_params["num_steps"], double_dqn=True)
        _loss = nn.MSELoss()(qs, target_qs)
        if self._teacher_params["supervise_loss"]["enabled"]:
            _loss += dqn.utils.train.supervise_loss(
                qs=qs, qs_full=qs_full, actions=actions,
                margin=self._teacher_params["supervise_loss"]["margin"],
                gpu=self._gpu_params["enabled"],
                gpu_id=self._gpu_params["id"],
                gpu_async=self._gpu_params["async"],
                loss_lambda=self._teacher_params["supervise_loss"]["lambda"],
                loss_type=self._teacher_params["supervise_loss"]["type"],
                learner_bs=0).mean()
        self.zero_grad()
        _loss.backward()
        return activations


    def _compute_grads_norm(self, bs):
        """D: compute L2 norm, but only for cosine progress measures.
        """
        _grad_norm = np.zeros(bs)
        for i in range(self._resizing_layer_idx, len(self.lower_nodes)):
            if len(self.upper_grads[i+1].shape) > 2:
                _z_grads = self.upper_grads[i+1].pow(2).sum(2).sum(2).sum(1)
            else:
                _z_grads = self.upper_grads[i+1].pow(2).sum(1)
            if len(self.upper_nodes[i].shape) > 2:
                _h = self.upper_nodes[i].pow(2).sum(2).sum(2).sum(1)
            else:
                _h = self.upper_nodes[i].pow(2).sum(1)
            _w_norm = ((_z_grads * _h).sqrt() * bs).pow(2)
            _b_norm = (_z_grads.sqrt() * bs).pow(2)
            _grad_norm += _w_norm.data.cpu().numpy() + _b_norm.data.cpu()\
                .numpy()
        return np.sqrt(_grad_norm)


    def _progress_computation(self, bs):
        """D: called from `self.progress_scores`, w/`bs` the batch size.

        Do matrix multiply with `torch.mm(mat1,mat2,out=None)` -> Tensor, where
        matrix multiplies are based on upper nodes/grads and lower nodes as
        computed in `self._progress_backwards`.

        The `_grad_changes` are directly the progress scores, since Allen's
        write-up shows (see Equation 7 and related discussion) that the diagonal
        of the `X * (dL/dX - dL/dX_0)` is equivalent.
        """
        _grad_changes = np.zeros(bs)
        if self._teacher_params["progress_measure"] == "cosine":
            _range = range(self._resizing_layer_idx, len(self.lower_nodes))
        else:
            _range = range(len(self.lower_nodes))
        for i in _range:
            _w = torch.diag(torch.mm(
                self.upper_nodes[i].view(bs, -1),
                (self.upper_grads[i] -
                 self.lower_nodes[i].grad).view(bs, -1).t())
            ) * bs   # TODO: why multiply by batch size? And similarly later?
            if len(self.upper_grads[i+1].shape) > 2:
                _b_grads = self.upper_grads[i+1].sum(2).sum(2)
            else:
                _b_grads = self.upper_grads[i+1]
            _b = torch.mm(
                _b_grads.data.view((bs, -1)),
                self.bias_diff[i+1].view((-1, 1))
            ) * bs
            _grad_changes += _w.data.cpu().numpy() + _b.view(-1).cpu().numpy()
        return _grad_changes


    def progress_scores(self, transitions, num_steps=None, activation=False):
        """Called via SnapshotsTeacher.generate_progress_scores(transitions).

        Only happens if the student's average episode reward is higher than the
        current matched teacher snapshot, so we need a new teacher snapshot.

        Start by going through all layers and computing the L2 norm. This is
        ||w-w*||_2 where w is current matched _teacher_ snapshot, and w* is the
        optimal (well, the last snapshot, but that's usually optimal) of the
        teacher. It's the denominator of the progress scores. (Edit: actually, w
        is the snapshot that's used for ZPD ... so really it's a few steps
        'ahead' of the new matched snapshot ...)

        Then, iterate through all elements in the teacher's replay buffer.

        Parameters
        ----------
        transitions: A set of transitions in general, but we actually pass the
            ENTIRE teacher's replay buffer, so we can get all its loaded
            transitions. Then use `get_transitions` by explicitly calling all
            the possible indices.
        num_steps: `n` in `n-step` returns to use for replay buffer sampling.
        activation: Boolean, if True add `activations` key to return value.
            Not sure why we have this, it doesn't affect code logic otherwise.

        Returns
        -------
        _return_info: Dictionary with different keys that we use for training
            later. Gets assigned to `replay.info` for sampling from the
            (teacher's) R-Buffer.
        Keys:
            - progress_scores: scores before we turn to a distribution by
                dealing with negative values and normalizing sum to one.
            - _progress_scores: after fixing negatives in `progress_scores`, but
                before we make it it sum to one.
            - p: After fixing `_progress_scores` to sum to one via softmax. Most
                important, vector of sampling (i.e., priority) probabilities for
                _all_ samples in the teacher's replay buffer.  We optionally
                smooth distribution with temperature, but I think this will give
                us nightmares. Keep it at 1.
            - progress_signs: `np.sign(...)` of progress_scores.
            - activations: From `self._progress_backwards`.
        """
        assert not self._teacher_params["teacher_samples_uniform"]
        _w_norm = 0.0
        if self._teacher_params["progress_measure"] == "cosine":
            _range = range(self._resizing_layer_idx, len(self._layers))
        else:
            _range = range(len(self._layers))
        for i in _range:
            w, w_star = self._layers[i][0], self._layers[i][1]
            _w_diff = (w.weight.data - w_star.weight.data).view(-1).pow(2).sum()
            _b_diff = (w.bias.data - w_star.bias.data).view(-1).pow(2).sum()
            _w_norm += _w_diff + _b_diff
        _w_norm = np.sqrt(_w_norm)
        self.zero_lower_bias()
        _num_iter = int(math.ceil(len(transitions) / self._teacher_params[
            "batch_size"]))
        _return_info = {
            "_progress_scores": [],
            "progress_scores": [],
            "progress_signs": [],
            "activations": []
        }
        for k in range(_num_iter):
            start_idx = k * self._teacher_params["batch_size"]
            end_idx = min((k + 1) * self._teacher_params["batch_size"],
                          len(transitions))
            _transitions = transitions.get_transitions(
                range(start_idx, end_idx), num_steps=num_steps)
            _activations = self._progress_backward(_transitions)
            if activation:
                _return_info["activations"].append(_activations.data
                                                   .cpu().numpy())
            _batch_scores = self._progress_computation(len(_transitions))
            if self._teacher_params["progress_measure"] == "cosine":
                _g_norm = self._compute_grads_norm(len(_transitions))
                _g_norm = _w_norm * _g_norm
                _batch_scores /= _g_norm
            else:
                _batch_scores /= _w_norm
            _return_info["progress_scores"].append(_batch_scores.copy())
            _return_info["progress_signs"].append(np.sign(_batch_scores))
            if not self._teacher_params["negative_correction"]:
                _batch_scores[_batch_scores <= 0] = self._teacher_params[
                    "progress_epsilon"]
            else:
                _batch_scores = np.abs(_batch_scores)
            _return_info["_progress_scores"].append(_batch_scores.copy())
        for keys, items in _return_info.items():
            if isinstance(items, list) and len(items) > 0:
                _return_info[keys] = np.concatenate(items)
        _progress_scores = _return_info["_progress_scores"]
        _return_info["p"] = convert_to_dist(
            _progress_scores, T=self._teacher_params["temperature"])
        self.revert_lower_bias()
        return _return_info


    def _compute_naive_progress_scores(self, transitions):
        """D: use for debugging.
        """
        if transitions is not None:
            assert len(transitions) == 1
            self._progress_backward(transitions, naive=True)
        assert len(self.lower_bias) == 0
        _progress_score = np.zeros((1, len(self._layers), 2))
        for i in range(len(self._layers)):
            w, w_star = self._layers[i][0], self._layers[i][1]
            _w_diff = (w.weight.data - w_star.weight.data).view(-1)
            _w_grad = w.weight.grad.data.view(-1)
            _progress_score[0, i, 0] = (_w_diff * _w_grad).sum()
            _b_diff = (w.bias.data - w_star.bias.data).view(-1)
            _b_grad = w.bias.grad.data.view(-1)
            _progress_score[0, i, 1] = (_b_diff * _b_grad).sum()
        return _progress_score


    def _compute_naive_grads_norm(self, transitions):
        """D: use for debugging.
        """
        if transitions is not None:
            assert len(transitions) == 1
            self._progress_backward(transitions, naive=True)
        assert len(self.lower_bias) == 0
        _grad_norm = np.zeros((1, len(self._layers), 2))
        for i in range(len(self._layers)):
            w = self._layers[i][0]
            _w_grad = w.weight.grad.norm()
            _b_grad = w.bias.grad.norm()
            _grad_norm[0, i, 0] = _w_grad
            _grad_norm[0, i, 1] = _b_grad
        return _grad_norm


    def _compute_weight_diff_norm(self):
        """D: use for debugging.
        """
        assert len(self.lower_bias) == 0
        _weight_norm = 0.0
        for layer in self._layers:
            w, w_star = layer[0], layer[1]
            _w_norm = (w.weight - w_star.weight).norm().pow(2)
            _b_norm = (w.bias - w_star.bias).norm().pow(2)
            _weight_norm += _w_norm + _b_norm
        return _weight_norm.sqrt().data.cpu().numpy()
