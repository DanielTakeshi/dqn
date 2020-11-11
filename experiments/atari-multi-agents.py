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
from dqn.utils.io import create_output_dir, write_dict, create_sub_dir
from dqn.utils.train import init_atari_model
from dqn.utils.logger import setup_logger
# from dqn.train import play, test
from dqn.processes import play_profiler as play
from dqn.environment.atari_wrappers import make_env
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from dqn.replay.experience import ExperienceReplay
from dqn.optimizer import Optimizer
from dqn.agent.dqn import DQNTrainAgent
from dqn.utils.debug import generate_debug_msg
from dqn.policy import GreedyEpsilonPolicy
from dqn.teacher.snapshots_teacher import SnapshotsReviewer
from dqn.teacher.teacher_centers import MultiAgentTeachingCenter
import cProfile
from pprint import pformat
import numpy as np


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", type=str,
                        help="Name of the experiment to run")
    parser.add_argument("--global-setting", type=str, 
                        default="global_settings",
                        help="Path of global configuration file")
    parser.add_argument("--seed", type=int, default="2",
                        help="Seed")
    params = parser.parse_args()
    params = Configurations(
        global_filename=params.global_setting,
        exp_name=params.exp,
        seed=params.seed)
    return params


def main_profiler(params):
    cProfile.runctx(
        'main(params)',
        globals(), locals(),
        os.path.join(params.params["log"]["dir"], 'main.prof'))


def init_learner(params, exp_queues, end_signals, learner_id):
    # Set up seeds
    params = params.copy()
    params["seed"] += learner_id
    params["log"]["dir"] = os.path.join(params["log"]["dir"],
                                        "learner_{0}".format(learner_id))
    create_sub_dir(params)
    set_all_seeds(seed=params["seed"], gpu=params["gpu"]["enabled"])
    setup_logger(dir_path=params["log"]["dir"],
                 filename="root",
                 level=params["log"]["log_level"])
    logger = logging.getLogger("learner_{0}".format(learner_id))

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
        gamma=params["train"]["gamma"], tag="learner_{0}".format(learner_id),
        debug_dir=os.path.join(params["log"]["dir"], "replay"))

    # Initialized optimizer
    opt = Optimizer(net=model, opt_params=params["opt"],
                    max_num_steps=params["env"]["max_num_steps"],
                    train_freq_per_step=params["train"]["train_freq_per_step"])

    # Initialize dqn agent.
    _policy = GreedyEpsilonPolicy(
        params=params["epsilon"], num_actions=params["env"]["num_actions"],
        max_num_steps=params["env"]["max_num_steps"],
        train_freq_per_step=params["train"]["train_freq_per_step"])

    reviewer = SnapshotsReviewer(
        writer=writer,
        model_name="{0}_({1})".format(params["exp"], params["seed"]-learner_id),
        teacher_params=params["multi_agent"],
        train_params=params["train"],
        gpu_params=params["gpu"],
        env_params=params["env"],
        log_params=params["log"],
        debug_dir=os.path.join(params["log"]["dir"], "review"),
        exp_queue=exp_queues[learner_id],
        learner_id=learner_id,
        opt_params=params["opt"])

    ts = MultiAgentTeachingCenter(
        writer=writer,
        exp_queues=exp_queues,
        end_signals=end_signals,
        learner_id=learner_id,
        log_params=params["log"],
        env_params=params["env"],
        teacher_params=params["multi_agent"],
        train_params=params["train"],
        gpu_params=params["gpu"])

    agent = DQNTrainAgent(net=model, gpu_params=params['gpu'],
                          log_params=params["log"],
                          opt=opt, train_params=params["train"],
                          replay=replay_memory, policy=_policy, teacher=ts,
                          avg_window=params["env"]["avg_window"],
                          tag="_{0}".format(learner_id))

    play_stop_flag = mp.Event()
    _agent_exp_queue = mp.Queue(
        maxsize=params["train"]['train_freq_per_step'] * 2)
    _play_process = mp.Process(
        target=play, args=(
            model, _agent_exp_queue, play_stop_flag, params["seed"],
            params["log"], params["gpu"], params["epsilon"], params["env"],
            params["train"]["train_freq_per_step"], "_{0}".format(learner_id)))
    _play_process.start()

    snapshots_summary = {}
    _play_rewards = []
    _play_speeds = []
    _timer = {}
    steps = 0
    snapshot_number = 1
    while True:
        for _ in range(params["train"]['train_freq_per_step']):
            steps += 1
            exp = _agent_exp_queue.get()
            if exp is None:
                _play_process.join()
                end_signals[learner_id].set()
                break
            replay_memory.add_one_transition(exp[0])
            if exp[1] is not None:
                if snapshot_number == 1:
                    agent.save_model(snapshot_number)
                    snapshots_summary[snapshot_number] = {
                        "rew": exp[1],
                        "steps": steps,
                    }
                    snapshot_number += 1
                _play_rewards.append(exp[1])
                _play_speeds.append(exp[2])

        if len(replay_memory) < params["replay"]['initial']:
            continue

        if end_signals[learner_id].is_set():
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
            if snapshot_number > 2:
                reviewer.matching(
                    np.mean(_play_rewards[-params["env"]["avg_window"]:]),
                    steps=steps)
                _timer[steps] = reviewer.timer()
                reviewer.reset_timer()
                write_dict(dict_object=_timer,
                           dir_path=params["log"]["dir"],
                           file_name="upload_summary")

        if steps % params["log"]["debug_per_step"] < \
                params["train"]['train_freq_per_step']:
            logger.info(generate_debug_msg(
                writer=writer, steps=steps, play_rewards=_play_rewards,
                play_speeds=_play_speeds, test_rewards=None,
                test_speeds=None, replay_memory=replay_memory,
                max_num_steps=params["env"]["max_num_steps"],
                agent=agent, avg_window=params["env"]["avg_window"]))


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

    exp_queues = [
        mp.Queue(maxsize=params["multi_agent"]["num_shared_exp"]) for _ in
        range(params["multi_agent"]["count"])]
    end_signals = [mp.Event() for _ in range(params["multi_agent"]["count"])]
    learners = [mp.Process(target=init_learner,
                           args=(params, exp_queues, end_signals, learner_id))
                for learner_id in range(params["multi_agent"]["count"])]

    for learner in learners:
        learner.start()

    while len(learners) > 0:
        removed_idx = []
        for i in range(len(learners)):
            if not learners[i].is_alive():
                learners[i].join()
                removed_idx.append(i)
        for i in removed_idx:
            del learners[i]

    logger.info("All learner finishes training!")


if __name__ == '__main__':
    mp = mp.get_context('spawn')
    # mp.set_start_method('spawn')
    # Read configurations from file
    _params = read_args()
    # Include log directory names in the params
    create_output_dir(params=_params.params)
    main_profiler(_params)
    # main(_params)
