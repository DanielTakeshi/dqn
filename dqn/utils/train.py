from dqn.utils.variables import create_var
import numpy as np
from scipy import signal
from dqn.models import AtariNet, AtariProgressNet
from dqn.utils.variables import cuda
from torch import nn


def get_qs_target(net, target_net, next_states, rewards, dones,
                  gamma, num_steps, double_dqn):
    """D: compute Q-values from the target network, for DQN.
    
    Called during normal DQN and during progres score computation if running
    student-teacher training.  If DQN, net and target_net have intuitive
    meanings. But for progress scores, they are both the same progress net?

    Basically this means there's no notion of a 'target net' in progress score.
    The target net is only in normal DQN to provide stability to training. I
    suppose this is OK for teachers?

    I _think_ the reason for detaching is that we already accumulated gradients
    from an earlier call in progress scores with the forward pass (_not_ the
    half-forward here) and we don't want to put more gradients in any of the
    upper layer nodes.

    The PyTorch tutorial also detaches.
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """
    if isinstance(target_net, AtariProgressNet):
        next_qs = target_net.half_forward(next_states, upper=True,
                                          activation=False).detach()
    else:
        next_qs = target_net(next_states).detach()

    if double_dqn:
        if isinstance(net, AtariProgressNet):
            next_actions = net.half_forward(next_states, upper=True,
                                            activation=False).detach()
        else:
            next_actions = net(next_states).detach()
        next_actions = next_actions.max(dim=1)[1]
        next_qs = next_qs.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_qs = next_qs.max(dim=1)[0]

    next_qs *= (1 - dones)

    # Commenting this out, in PyTorch 0.4.1 it's *always* False.
    #next_qs.volatile = False

    return rewards + next_qs * (gamma ** num_steps)


def get_qs(net, states, actions):
    """D: compute Q-values from `bellman_loss`.

    Returns
    -------
    (Q-values of actions chosen, Q-values of all actions)
        E.g., in Pong, shapes would be (MB,) and (MB,6), respectively.
    """
    qs = net(states)
    return qs.gather(1, actions.unsqueeze(-1)).squeeze(-1), qs


def get_qs_(net, states, actions, upper_forward=False):
    """D: similar to `get_qs_`, but called from progress net code.

    In progress score computation, `upper_forward=False` by default, so it's the
    normal net forward method, not the half forward, though the progress net has
    a complicated forward method. We also don't do activations here as that's
    the job of `get_activation` (see below).

    Returns
    -------
    (Q-values of actions chosen, Q-values of all actions)
        E.g., in Pong, shapes would be (MB,) and (MB,6), respectively.
    """
    if upper_forward:
        assert isinstance(net, AtariProgressNet)
        qs = net.half_forward(states, activation=False, upper=True)
    else:
        qs = net.forward(states, activation=False)
    return qs.gather(1, actions.unsqueeze(-1)).squeeze(-1), qs


def get_activation(net, states, upper=False):
    """D: called from progress scores after `get_qs_`.

    Now, we _do_ run the 'half forward' method, the `upper` part (it's set as
    True in the progress score computation). We already got the Q-values, so we
    ignore that part of the argument and just return activations. Since it's the
    progress net, the half_forward returns the input to the _last_ FC layer.
    """
    if isinstance(net, AtariProgressNet):
        _, activations = net.half_forward(states, activation=True,
                                          upper=upper)
    else:
        _, activations = net.forward(states, activation=True)
    return activations


def bellman_loss(net, target_net, loss, states, next_states, actions, rewards,
                 dones, gamma, num_steps, double_dqn):
    """Called from `agent/dqn.py` (and progress net), for Bellman loss.
    """
    qs, qs_full = get_qs(net, states, actions)
    target_qs = get_qs_target(net, target_net, next_states, rewards, dones,
                              gamma, num_steps, double_dqn)
    return loss(qs, target_qs), qs, qs_full


def margin_loss(qs, qs_full, actions, margin, gpu, gpu_id, gpu_async):
    """Also known as the supervised loss, or imitation loss, from DQfD.

    See https://github.com/CannyLab/dqn/issues/19 for details. The `actions`
    select rows from `margins` which are added to `q_full`.

    HUGE NOTE: Allen was squaring the end result. I don't think we want that.

    Parameters
    ----------
    qs: torch.cuda.FloatTensor
        Q-values of only selected actions in the minibatch, (MB,).
    qs_full: torch.cuda.FloatTensor
        Q-values of actions in all minibatch elements, (MB,num_actions).
    actions: torch.cuda.LongTensor
        Tensor of shape (MB,) representing index of actions selected.
    margin: float
        The large margin value, should usually be 0.8 (it's from DQfD).

    Returns
    -------
    Large margin loss, but for each minibatch element. We clear out the
    learner's minibatch later as those should have loss 0.
    """
    num_actions = qs_full.size(1)
    margins = create_var(num_actions, gpu=gpu, gpu_id=gpu_id,
                         gpu_async=gpu_async,
                         convert_func="torch_ones_minus_eye")
    margins *= margin
    margins = margins[actions.detach()]
    qs_max = (qs_full + margins).max(dim=1)[0]
    #return (qs_max - qs).pow(2)  # Daniel: ???????????
    return (qs_max - qs)


def cross_entropy_loss(qs_full, actions):
    _loss = nn.CrossEntropyLoss(reduce=False)(qs_full, actions)
    return _loss


def supervise_loss(qs, qs_full, actions, margin, gpu, gpu_id, gpu_async,
                   loss_lambda, loss_type, learner_bs=0):
    """Called from `agent/dqn.py`, for computing 'large margin' loss from DQfD.
    
    We multiply loss_lambda here; keep in mind when plotting, though DeepMind
    used lambda=1.0 for the supervised loss. Learner samples come BEFORE teacher
    samples in the minibatch!

    Parameters
    ----------
    loss_type: str
        One of two loss types. Use margin, as DQfD showed that's slightly better
        than cross_entropy.
    learner_bs: int
        Number of samples in minibatch from the learner's own history. These are
        zero'd out since it's self-generated data.
    """
    if loss_type == "cross_entropy":
        _loss = cross_entropy_loss(qs_full=qs_full, actions=actions)
    else:
        assert loss_type == "margin"
        _loss = margin_loss(qs=qs, qs_full=qs_full, actions=actions,
                            margin=margin, gpu=gpu, gpu_id=gpu_id,
                            gpu_async=gpu_async)
    _loss.data[0: learner_bs] *= 0.0
    return _loss * loss_lambda


def merge_transitions(learner, teacher):
    if teacher is None:
        return learner
    elif learner is None:
        return teacher
    else:
        learner.state.extend(teacher.state)
        learner.action.extend(teacher.action)
        learner.reward.extend(teacher.reward)
        learner.done.extend(teacher.done)
        learner.next_state.extend(teacher.next_state)
        if isinstance(learner.weight, np.ndarray):
            learner.weight = np.concatenate((learner.weight, teacher.weight))
        else:
            learner.weight.extend(teacher.weight)
        return learner


def merge_weights(weights, learner_bs, teacher_bs):
    if teacher_bs == 0:
        return np.ones(learner_bs)
    assert len(weights) == teacher_bs
    return np.concatenate((np.ones(learner_bs), weights))


def compute_discount(x, gamma):
    """
    Compute discounted sum of future values
    out = x[0] + gamma * x[1] + gamma^2 * x[2] + ...

    :param x: input array/list
    :param gamma: decay rate
    :return an output to apply the discount rates
    """
    _output = 0.0
    for i in reversed(x):
        _output *= gamma
        _output += i
    return _output


def trajectory_discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = x[i] + gamma * x[i+1] + gamma^2 * x[i+2] + ...

    :param x: input array/list
    :param gamma: decay rate
    :return an output list to apply the discount rates
    """
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def init_atari_model(obs_space, num_actions, hidden_size, gpu, gpu_id):
    assert isinstance(gpu, bool)
    assert isinstance(gpu_id, int) and gpu_id >= 0
    assert isinstance(num_actions, int) and num_actions >= 0
    assert isinstance(hidden_size, int) and hidden_size >= 0
    assert isinstance(obs_space, list) or isinstance(obs_space, tuple)

    _model = AtariNet(obs_space=tuple(obs_space), num_actions=num_actions,
                      hidden_size=hidden_size)
    _model = cuda(x=_model, gpu=gpu, gpu_id=gpu_id)
    return _model


def init_atari_progress_model(obs_space, num_actions, train_params, gpu_params,
                              teacher_params, opt_params, max_num_steps):
    assert isinstance(num_actions, int) and num_actions >= 0
    assert isinstance(obs_space, list) or isinstance(obs_space, tuple)

    _model = AtariProgressNet(
        obs_space=tuple(obs_space), num_actions=num_actions,
        train_params=train_params, gpu_params=gpu_params,
        teacher_params=teacher_params, opt_params=opt_params,
        max_num_steps=max_num_steps)
    _model = cuda(x=_model, gpu=gpu_params["enabled"], gpu_id=gpu_params["id"])
    return _model
