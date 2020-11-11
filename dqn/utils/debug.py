import numpy as np


def generate_debug_msg(writer, steps, play_rewards, play_speeds, test_rewards,
                       test_speeds, replay_memory, max_num_steps, agent,
                       avg_window, true_results):
    """D: the messasge that we see on the command line showing the training
    process (and the testing, if we have a testing process).
    """
    _debug_msg = "\n********** {0} steps ({1:.2%}) completed " \
                 "**********\n" \
        .format(steps, float(steps) / max_num_steps)
    _debug_msg += performance_debug(writer, steps, play_rewards, play_speeds,
                                    avg_window, train=True)
    if test_rewards and len(test_rewards) > 0:
        _debug_msg += performance_debug(writer, steps, test_rewards,
                                        test_speeds, avg_window, train=False)
    _debug_msg += agent_debug(writer, steps, agent)
    _debug_msg += _generate_true_msg(true_results, avg_window)
    replay_memory.debug(steps)
    return _debug_msg


def _generate_true_msg(true_results, avg_w):
    """D: I added this for the unwrapped stuff.
    """
    rewards = true_results['r'].tolist()
    lengths = true_results['l'].tolist()
    _debug_msg = ">> True stuff\nEpisodes: {0}\n" \
                 "Last {1} results: avg {2:.2f}, std {3:.2f}, " \
                 "min {4:.2f}, max {5:.2f}\n".format(
            len(rewards), avg_w, np.mean(rewards[-avg_w:]),
            np.std(rewards[-avg_w:]), np.min(rewards[-avg_w:]),
            np.max(rewards[-avg_w:]))
    return _debug_msg


def performance_debug(writer, steps, rewards, speeds, avg_window, train):
    if train:
        flag = "Training"
    else:
        flag = "Testing"
    writer.add_scalar("reward/{0}".format(flag.lower()),
                      float(np.mean(rewards[-avg_window:])), steps)
    writer.add_scalar("speed/{0}".format(flag.lower()),
                      float(np.mean(speeds[-avg_window:])), steps)
    _debug_msg = ">> {0} Agent Summary \nTotal episodes: {1}\n" \
                 "Recent {2} episode performance: avg {3:.2f}, " \
                 "std {4:.2f}, min {5:.2f}, max {6:.2f} \n" \
                 "Average Recent {2} episode speed: {7:.2f} " \
                 "frames/seconds\n" \
        .format(
            flag, len(rewards), avg_window, np.mean(rewards[-avg_window:]),
            np.std(rewards[-avg_window:]), np.min(rewards[-avg_window:]),
            np.max(rewards[-avg_window:]), np.mean(speeds[-avg_window:]))
    return _debug_msg


def agent_debug(writer, steps, agent):
    debug_msg = ">> Training Summary \n"
    debug_msg += "exploration rates: {0:.2f}\n".format(
        agent.get_policy_epsilon(steps))
    writer.add_scalar("exploration_rates", agent.get_policy_epsilon(steps),
                      steps)
    debug_msg += "learning rates: {0:.5f}\n".format(
        agent.get_current_lr(steps))
    writer.add_scalar("learning_rates", agent.get_current_lr(steps), steps)
    for loss_label, loss_queue in agent.debug_loss.items():
        writer.add_scalar("loss/" + loss_label, loss_queue.mean(), steps)
        debug_msg += "{0}: {1:.5f}\n".format(loss_label.replace("_", " "),
                                             loss_queue.mean())
    return debug_msg
