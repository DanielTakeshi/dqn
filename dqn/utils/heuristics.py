"""
Use for heuristics, e.g., to help exit earlier.
A brand new file added by Daniel.
"""
import logging
import numpy as np


def perform_bad_exit_early(params, steps, play_rewards):
    """Determine if we should exit a training run early; could save time.

    Parameters
    ----------
    params: Parameters dict we use for training.
    steps: Number of steps in the training process.
    play_rewards: List of rewards from the training process.
    """
    if len(play_rewards) == 0:
        return False
    logger = logging.getLogger('root')
    past_100 = np.mean(play_rewards[-params["env"]["avg_window"]:])
    if "Pong" in params["env"]["name"] and steps > 1000000 and past_100 < -19.5:
        logger.info("ALERT! Exiting after {} steps due to rew: {:.1f}".format(
                steps, past_100))
        return True
    return False
