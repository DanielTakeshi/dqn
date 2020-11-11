import os
import sys
import inspect
_parent_dir = os.path.dirname(os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, _parent_dir)
import argparse
import logging
from dqn.configs import Configurations
from dqn.utils.setup import cuda_config, set_all_seeds
from dqn.utils.io import create_output_dir, write_dict
from dqn.utils.train import init_atari_model
from dqn.utils.logger import setup_logger
# from dqn.train import play, test
from dqn.processes import play_profiler as play, test_profiler as test
from dqn.environment.atari_wrappers import make_env
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from dqn.replay.experience import ExperienceReplay
from dqn.optimizer import Optimizer
from dqn.agent.dqn import DQNTrainAgent
from dqn.teacher.teacher_centers import SnapshotDistillation
from dqn.utils.debug import generate_debug_msg
from dqn.policy import GreedyEpsilonPolicy
import cProfile
from pprint import pformat
import numpy as np


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--id", type=str,
                        help="ID of the experiment to run")
    parser.add_argument("-e", "--exp", type=str,
                        default="settings",
                        help="Name of the experiment to run")
    parser.add_argument("-n", "--note", type=str, default="",
                        help="Change of hyper-parameters in training")
    params = parser.parse_args()
    params = Configurations(
        exp_name=params.exp,
        exp_id=params.id,
        note=params.note)
    return params


def main_profiler(params):
    cProfile.runctx(
        'main(params)',
        globals(), locals(),
        os.path.join(params.params["log"]["dir"], 'main.prof'))


def main(params):
    # Remap the gpu devices if using gpu
    if cuda_config(gpu=params.params["gpu"]["enabled"],
                   gpu_id=params.params["gpu"]["id"]):
        params.params["gpu"]["id"] = 0

    # Include log directory names in the params
    setup_logger(dir_path=params.params["log"]["dir"],
                 filename="root",
                 level=params.params["log"]["log_level"])
    logger = logging.getLogger("root")
    logger.info("Output directory is located at {0}."
                .format(params.params["log"]["dir"]))

    # Set up environment
    _env = make_env(
        env_name=params.params["env"]["name"],
        episode_life=params.params["env"]["episode_life"],
        skip_frame=params.params["env"]["frame_skip"],
        clip_rewards=params.params["env"]["clip_rewards"],
        frame_stack=params.params["env"]["frame_stack"])
    params.params["env"]["num_actions"] = _env.action_space.n
    params.params["env"]["action_meanings"] = \
        _env.unwrapped.get_action_meanings()
    params.params["env"]["obs_space"] = _env.observation_space.shape

    # Output all configurations
    params = params.dump()
    logger.debug("Hyper-parameters have been written to disk as followings:"
                 " {0}".format(pformat(params)))

    # Set up seeds
    set_all_seeds(seed=params["seed"], gpu=params["gpu"]["enabled"])

    # Initialize dqn Q-network
    model = init_atari_model(
        obs_space=params["env"]["obs_space"],
        num_actions=params["env"]["num_actions"],
        hidden_size=params["train"]["hidden_size"],
        gpu=params["gpu"]["enabled"],
        gpu_id=params["gpu"]["id"])
    model.init_weight()
    model.share_memory()

    # Initialize TensorBoard writer for visualization purposes
    writer = SummaryWriter(log_dir=params["log"]["dir"],
                           comment="-" + params["env"]['name'])

    # Initialized replay buffer
    replay_memory = ExperienceReplay(
        writer=writer,
        capacity=params["replay"]["size"],
        init_cap=params["replay"]["initial"],
        frame_stack=params["env"]["frame_stack"],
        gamma=params["train"]["gamma"], tag="train",
        debug_dir=os.path.join(params["log"]["dir"], "learner_replay"))

    # Initialized optimizer
    opt = Optimizer(net=model, opt_params=params["opt"],
                    max_num_steps=params["env"]["max_num_steps"],
                    train_freq_per_step=params["train"]["train_freq_per_step"])

    # Initialized teacher
    ts = SnapshotDistillation(
        writer=writer,
        teacher_params=params["teacher"],
        train_params=params["train"],
        gpu_params=params["gpu"],
        env_params=params["env"],
        log_params=params["log"],
        debug_dir=os.path.join(params["log"]["dir"], "teaching"),
        opt_params=params["opt"])

    # Initialize dqn agent.
    _policy = GreedyEpsilonPolicy(
        params=params["epsilon"], num_actions=params["env"]["num_actions"],
        max_num_steps=params["env"]["max_num_steps"],
        train_freq_per_step=params["train"]["train_freq_per_step"])
    agent = DQNTrainAgent(net=model, gpu_params=params['gpu'],
                          log_params=params["log"],
                          opt=opt, train_params=params["train"],
                          replay=replay_memory, policy=_policy, teacher=ts,
                          avg_window=params["env"]["avg_window"])

    # Initialized other processes
    exp_queue = mp.Queue(maxsize=params["train"]['train_freq_per_step'] * 2)
    play_stop_flag = mp.Event()
    _play_process = mp.Process(
        target=play, args=(
            model, exp_queue, play_stop_flag, params["seed"], params["log"],
            params["gpu"], params["epsilon"], params["env"],
            params["train"]["train_freq_per_step"]))
    _play_process.start()
    rew_queue = mp.Queue(maxsize=2)
    test_stop_flag = mp.Event()
    _test_process = mp.Process(
        target=test, args=(
            model, rew_queue, test_stop_flag, params["seed"], params["log"],
            params["gpu"], params["epsilon"], params["env"],
            params["train"]["train_freq_per_step"]))
    _test_process.start()

    snapshots_summary = {}
    snapshot_number = 1
    _play_rewards = []
    _play_speeds = []
    _test_rewards = []
    _test_speeds = []
    steps = 0
    _end = False
    while True:
        for _ in range(params["train"]['train_freq_per_step']):
            steps += 1
            exp = exp_queue.get()
            if exp is None:
                _play_process.join()
                _end = True
                break
            replay_memory.add_one_transition(exp[0])
            ts.read_n_transitions(n=1)
            if exp[1] is not None:
                _play_rewards.append(exp[1])
                _play_speeds.append(exp[2])
                test_rew = rew_queue.get()
                if test_rew is None:
                    _test_process.join()
                    _end = True
                    break
                _test_rewards.append(test_rew[0])
                _test_speeds.append(test_rew[1])

        if "Pong" in params["env"]["name"] and steps > 1000000 and \
                np.mean(_play_rewards[-params["env"]["avg_window"]:]) < -20:
            break

        if len(replay_memory) < params["replay"]['initial']:
            continue

        if _end:
            write_dict(dict_object=snapshots_summary,
                       dir_path=params["log"]["dir_snapshots"],
                       file_name="snapshots_summary")
            break

        agent.train(steps=steps)

        if steps % params["log"]['snapshot_per_step'] < \
                params["train"]['train_freq_per_step']:
            agent.save_model(snapshot_number)
            snapshots_summary[snapshot_number] = {
                "rew": np.mean(_play_rewards[-params["env"]["avg_window"]:]),
                "steps": steps,
            }
            snapshot_number += 1
            write_dict(dict_object=snapshots_summary,
                       dir_path=params["log"]["dir_snapshots"],
                       file_name="snapshots_summary")

        if steps % params["log"]["debug_per_step"] < \
                params["train"]['train_freq_per_step']:
            ts.matching(
                np.mean(_play_rewards[-params["env"]["avg_window"]:]),
                steps=steps)
            logger.info(generate_debug_msg(
                writer=writer, steps=steps, play_rewards=_play_rewards,
                play_speeds=_play_speeds, test_rewards=_test_rewards,
                test_speeds=_test_speeds, replay_memory=replay_memory,
                max_num_steps=params["env"]["max_num_steps"],
                agent=agent, avg_window=params["env"]["avg_window"]))

    if _play_process.is_alive():
        play_stop_flag.set()
        while exp_queue.get() is not None:
            pass
        _play_process.join()

    if _test_process.is_alive():
        test_stop_flag.set()
        while rew_queue.get() is not None:
            pass
        _test_process.join()

    logger.info("Training complete!")


if __name__ == '__main__':
    mp = mp.get_context('spawn')
    # mp.set_start_method('spawn')
    # Read configurations from file
    _params = read_args()
    # Include log directory names in the params
    main_profiler(_params)
    # main(_params)
