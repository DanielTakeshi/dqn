from dqn.replay.transition import Transition


def merge_transitions_xp(transitions):
    """
    Find multiple samples in the replay buffer and output a numpy array
    stacking all samples together.

    Changing name to have `_xp` to avoid confusion with the method defined
    in `utils/train.py`.

    :param transitions: A list of `Transition` objects.

    :return: A `Transition` object, where each field is a `numpy` array
    stacked all requested samples
    """
    _transitions = Transition(state=[], next_state=[], action=[],
                              reward=[], done=[], weight=[])
    for transition in transitions:
        _transitions.state.append(transition.state)
        _transitions.next_state.append(transition.next_state)
        _transitions.action.append(transition.action)
        _transitions.reward.append(transition.reward)
        _transitions.done.append(int(transition.done))
        _transitions.weight.append(transition.weight)
    return _transitions
