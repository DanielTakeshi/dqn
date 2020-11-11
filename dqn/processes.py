import logging
from dqn.utils.setup import set_all_seeds
from dqn.utils.data_structures import Tracker
from dqn.utils.io import write_dict
from dqn.utils.logger import setup_logger
from dqn.environment.setup import Environment
from dqn.agent.dqn import DQNAgent
from dqn.policy import GreedyEpsilonPolicy
from dqn.replay.experience import ExperienceSource
import cProfile
import os


def play_profiler(model, exp_queue, stop_flag, seed, log_params, gpu_params,
                  epsilon_params, env_params, train_freq_per_step, tag=""):
    """Run `play(...)`, potentially with profiling.

    Note that our `experiments/atari.py` refers to this (and not `play`).
    Put results in `play.prof`, which is separate from the `main.prof`,
    presumably due to different processes.  You can enable or disable by
    commenting out the appropriate method.
    """
    cProfile.runctx(
        'play(model, exp_queue, stop_flag, seed, log_params, gpu_params, '
        'epsilon_params, env_params, train_freq_per_step, tag)',
        globals(), locals(),
        os.path.join(log_params["dir"], 'play.prof'))


def test_profiler(model, reward_queue, stop_flag, seed, log_params, gpu_params,
                  epsilon_params, env_params, train_freq_per_step, tag=""):
    """Similar to `play_profiler` except for 'test-set' rollouts.

    I.e., those rollouts used for evaluation but not for gradient updates.  Put
    results in `test.prof`. I don't use this since it seems un-necessary.
    """
    cProfile.runctx(
        'test(model, reward_queue, stop_flag, seed, log_params, '
        'gpu_params, epsilon_params, env_params, train_freq_per_step, tag)',
        globals(), locals(),
        os.path.join(log_params["dir"], 'test.prof'))


def play(model, exp_queue, stop_flag, seed, log_params, gpu_params,
         epsilon_params, env_params, train_freq_per_step, tag=""):
    """Execute training.

    Here we have `exp_queue` which `atari.py` uses (via `.get()`) so that it
    obtains elements needed for the replay buffer. We always put in 4-tuples
    into the queue (and by extension the replay buffer) but unless there is a
    reward, the last three items are None. When there is a reward we add more
    info to the other three tuple elements.

    We also save the two summaries in the episodes directory. The true summary
    at v02 has extra information about lives.

    I assume that because `model` is passed as input, that the agent here
    (`agent`) is always synchronized with the agent from `atari.py`.
    """
    setup_logger(dir_path=log_params["dir"], filename="train",
                 level=log_params["log_level"])
    logger = logging.getLogger("train_env{0}".format(tag))
    logger.info("Training environment has been connected and set up.")

    # Set up seeds
    set_all_seeds(seed=seed, gpu=gpu_params["enabled"])

    # Initialize environment
    train_env = Environment(env_params=env_params, log_params=log_params,
                            train=True, logger=logger, seed=seed)
    train_env.env._max_episode_steps = 18000
    policy = GreedyEpsilonPolicy(
        params=epsilon_params, num_actions=env_params["num_actions"],
        max_num_steps=env_params["max_num_steps"],
        train_freq_per_step=train_freq_per_step)
    agent = DQNAgent(net=model, gpu_params=gpu_params,
                     log_params=log_params, policy=policy)

    exp_source = ExperienceSource(env=train_env, agent=agent,
                                  episode_per_epi=log_params["episode_per_epi"])
    exp_source_iter = iter(exp_source)

    if tag == "":
        track = Tracker(max_num_steps=env_params["max_num_steps"],
                        logger=logger)
    else:
        track = Tracker(stop_reward=env_params["stop_reward"], logger=logger)
    with track as reward_tracker:
        while True:
            exp = next(exp_source_iter)
            rewards, mean_rewards, speed = exp_source.pop_latest()
            if rewards is not None:
                _end = reward_tracker.reward(
                    mean_rewards=mean_rewards,
                    frame=train_env.total_steps)
                exp_queue.put((exp, rewards, speed, mean_rewards))
                write_dict(dict_object=train_env.summary,
                           dir_path=log_params["dir_episodes"],
                           file_name="training_summary_lifes_clip")
                write_dict(dict_object=train_env.summary_true,
                           dir_path=log_params["dir_episodes"],
                           file_name="training_summary_true_v02")
                if _end:
                    break
            else:
                exp_queue.put((exp, None, None, None))
            if stop_flag.is_set():
                break
    exp_queue.put(None)


def test(model, reward_queue, stop_flag, seed, log_params, gpu_params,
         epsilon_params, env_params, train_freq_per_step, tag=""):
    """D: for the testing process.

    Now the queue that we use needs 3-tuple elements. Each tuple has reward,
    speed, and total steps.
    """
    setup_logger(dir_path=log_params["dir"], filename="test",
                 level=log_params["log_level"])
    logger = logging.getLogger("test_env{0}".format(tag))
    logger.info("Testing environment has been connected and set up.")

    # Set up seeds
    set_all_seeds(seed=seed, gpu=gpu_params["enabled"])
    # Initialize environment
    test_env = Environment(env_params=env_params, log_params=log_params,
                           train=False, logger=logger, seed=seed)
    test_env.env._max_episode_steps = 18000
    policy = GreedyEpsilonPolicy(
        params=epsilon_params, num_actions=env_params["num_actions"],
        max_num_steps=env_params["max_num_steps"],
        train_freq_per_step=train_freq_per_step)
    agent = DQNAgent(net=model, gpu_params=gpu_params,
                     log_params=log_params, policy=policy)
    logger.info("Initialized DQN agent and policy selector.")

    exp_source = ExperienceSource(env=test_env, agent=agent,
                                  episode_per_epi=log_params["episode_per_epi"])
    exp_source_iter = iter(exp_source)

    if env_params['stop_reward']:
        tracker = Tracker(stop_reward=env_params['stop_reward'], logger=logger)
    else:
        tracker = Tracker(max_num_steps=env_params["max_num_steps"] * 100,
                          logger=logger)
    with tracker as reward_tracker:
        while True:
            _ = next(exp_source_iter)
            rewards, mean_rewards, speed = exp_source.pop_latest()
            if rewards is not None:
                _end = reward_tracker.reward(
                    mean_rewards=mean_rewards,
                    frame=test_env.total_steps)
                reward_queue.put((rewards, speed, test_env.total_steps))
                write_dict(dict_object=test_env.summary,
                           dir_path=log_params["dir_episodes"],
                           file_name="testing_summary")
                if _end:
                    break
            if stop_flag.is_set():
                break
    reward_queue.put(None)
