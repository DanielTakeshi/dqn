import os
import argparse
import logging
import time
from dqn.configs import Configurations
from dqn.utils.setup import cuda_config, set_all_seeds
from dqn.utils.io import write_dict
from dqn.utils.train import init_atari_model
from dqn.utils.logger import setup_logger
from dqn.environment.atari_wrappers import make_env
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from dqn.replay.experience import ExperienceReplay
from dqn.optimizer import Optimizer
from dqn.agent.dqn import DQNTrainAgent
from dqn.teacher.teacher_centers import MultiTeacherTeachingCenter
from dqn.utils.debug import generate_debug_msg
from dqn.policy import GreedyEpsilonPolicy
import cProfile
from pprint import pformat
import numpy as np
from dqn.utils.heuristics import perform_bad_exit_early
from dqn.environment import monitor


def main_profiler(params):
    """
    A similar 'roundabout' procedure is done in training when calling the play
    and test methods: it is actually called through the cProfile profiler.
    """
    pth = os.path.join(params.params["log"]["dir"], 'main.prof')
    cProfile.runctx('main(params)', globals(), locals(), pth)


def get_true_rew(monitor_dir):
    true_results = monitor.load_results(monitor_dir)
    true_rews = true_results['r'].tolist()
    return true_rews


def main(params):
    if params.profile:
        from dqn.processes import play_profiler as play, test_profiler as test
    else:
        from dqn.processes import play, test
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

    # Set env to add stuff to `params['env']` -- we do NOT use this `_env` for
    # stepping; we create again in `dqn/processes.py` via `Environment` class.
    _env = make_env(
        env_name=params.params["env"]["name"],
        episode_life=params.params["env"]["episode_life"],
        skip_frame=params.params["env"]["frame_skip"],
        clip_rewards=params.params["env"]["clip_rewards"],
        frame_stack=params.params["env"]["frame_stack"],
        logdir=params.params["log"]["dir"])
    params.params["env"]["num_actions"] = _env.action_space.n
    params.params["env"]["action_meanings"] = \
        _env.unwrapped.get_action_meanings()
    params.params["env"]["obs_space"] = _env.observation_space.shape
    params.params["env"]["total_lives"] = _env.unwrapped.ale.lives()

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

    # Initialized replay buffer. Not quite like OpenAI, seems to operate at
    # another 'granularity'; that of episodes, in addition to transitions?
    replay_memory = ExperienceReplay(
        writer=writer,
        capacity=params["replay"]["size"],
        init_cap=params["replay"]["initial"],
        frame_stack=params["env"]["frame_stack"],
        gamma=params["train"]["gamma"], tag="train",
        debug_dir=os.path.join(params["log"]["dir"], "learner_replay"))

    # Initialized optimizer with decaying `lr_schedule` like OpenAI does.
    opt = Optimizer(net=model, opt_params=params["opt"],
                    max_num_steps=params["env"]["max_num_steps"],
                    train_freq_per_step=params["train"]["train_freq_per_step"])

    # Initialized teacher. New thing, supports _ensemble_ of teachers. But if
    # our params are from a non-student/teacher run, e.g., `pong_standard_fast`,
    # then there are no teacher params.
    ts = MultiTeacherTeachingCenter(
        writer=writer,
        teacher_params=params["teacher"] if "teacher" in params else None,
        train_params=params["train"],
        gpu_params=params["gpu"],
        env_params=params["env"],
        log_params=params["log"],
        debug_dir=os.path.join(params["log"]["dir"], "teaching"),
        opt_params=params["opt"])

    # Initialize dqn agent. Like I do, it uses a schedule.
    _policy = GreedyEpsilonPolicy(
        params=params["epsilon"], num_actions=params["env"]["num_actions"],
        max_num_steps=params["env"]["max_num_steps"],
        train_freq_per_step=params["train"]["train_freq_per_step"])
    agent = DQNTrainAgent(net=model, gpu_params=params['gpu'],
                          log_params=params["log"],
                          opt=opt, train_params=params["train"],
                          replay=replay_memory, policy=_policy, teacher=ts,
                          avg_window=params["env"]["avg_window"])

    # --------------------------------------------------------------------------
    # Initialized other processes. From torch.multiprocessing, 100% compatible
    # with normal python multiprocessing, so look at docs there.  E.g., create
    # `Process` w/target and args, and start w/[proc].start(). We have play and
    # test processes. We get a process each for training and testing; debug logs
    # show that train/test envs happen simultaneously.
    # --------------------------------------------------------------------------
    exp_queue = mp.Queue(maxsize=params["train"]['train_freq_per_step'] * 2)
    play_stop_flag = mp.Event()
    _play_process = mp.Process(
        target=play, args=(
            model, exp_queue, play_stop_flag, params["seed"], params["log"],
            params["gpu"], params["epsilon"], params["env"],
            params["train"]["train_freq_per_step"]))
    _play_process.start()

    # Similarly, the test process, which gets `rew_queue` not `exp_queue`.
    have_test_proc = params['log']['have_test_proc']
    if have_test_proc:
        rew_queue = mp.Queue(maxsize=2)
        test_stop_flag = mp.Event()
        _test_process = mp.Process(
            target=test, args=(
                model, rew_queue, test_stop_flag, params["seed"], params["log"],
                params["gpu"], params["epsilon"], params["env"],
                params["train"]["train_freq_per_step"]))
        _test_process.start()

    # Finally, training. See `global_settings` and other experiment files.
    snapshots_summary = {}
    info_summary = {}
    snapshot_number = 1
    _play_rewards = []  # clipped rewards
    _play_speeds = []
    _test_rewards = []  # clipped rewards
    _test_speeds = []
    steps = 0
    _end = False
    num_true_episodes = 0
    time_start = time.time()
    avg_w = params["env"]["avg_window"]
    matched_list = None
    test_match = False

    while True:
        # ----------------------------------------------------------------------
        # Go through sufficient steps before training, default is 4, similar to
        # my (serial) Pong code. Also, note that we deal with `steps` here, but
        # each `step` is actually 4 'frames' in Pong emulator. I think Allen's
        # schedules, though params say 'frames', are actually wrt steps.  Get
        # from `exp_queue`, not `rew_queue`, presumably because we should use
        # samples from the former process for training. Queue returns `exp`:
        #     (<dqn.replay.transition.Transition, None, None, None)
        # If nothing left, we're done.
        # ----------------------------------------------------------------------
        for _ in range(params["train"]['train_freq_per_step']):
            steps += 1
            exp = exp_queue.get()
            if exp is None:
                _play_process.join()
                _end = True
                break
            replay_memory.add_one_transition(exp[0])
            ts.read_n_transitions(n=1)

            # ------------------------------------------------------------------
            # Only invokes if more in exp, happens when finishing a _life_.  If
            # no teachers, `ts.matching` (needs true rew) does nothing. I
            # changed so we only do _matching_ check if an _episode_ finished;
            # you have to finish a life before an episode can possibly be done.
            # ------------------------------------------------------------------
            if exp[1] is not None:
                _play_rewards.append(exp[1])
                _play_speeds.append(exp[2])

                rew_list = get_true_rew(params["log"]["dir"])
                if len(rew_list) > num_true_episodes:
                    assert len(rew_list) == num_true_episodes + 1
                    num_true_episodes += 1
                    if len(replay_memory) >= params["replay"]['initial']:
                        rew_list = get_true_rew(params["log"]["dir"])
                        rew_true = np.mean(rew_list[-avg_w:])
                        matched_list = ts.matching(rew_true, steps=steps, agent=agent)
                        if matched_list is not None:
                            test_match = True

                if have_test_proc:
                    test_rew = rew_queue.get()
                    if test_rew is None:
                        _test_process.join()
                        _end = True
                        break
                    _test_rewards.append(test_rew[0])
                    _test_speeds.append(test_rew[1])

        # Exit early, ignore training, or finish training.
        if perform_bad_exit_early(params, steps, _play_rewards):
            break
        if len(replay_memory) < params["replay"]['initial']:
            continue
        if _end:
            write_dict(dict_object=snapshots_summary,
                       dir_path=params["log"]["dir_snapshots"],
                       file_name="snapshots_summary")
            break

        # For teacher-only, if matched to something. Save snapshot. We can use
        # to check if saved snapshots 'match' the teacher snapshot.  The other
        # snapshot saving is for later with training a single teacher.
        if test_match:
            assert matched_list is not None
            for tidx, titem in enumerate(matched_list):
                if titem is not None:
                    logger.info("matched_list (ZPD): {} w/{} idx, {} item".format(
                        matched_list, tidx, titem))
                    rew_list = get_true_rew(params["log"]["dir"])
                    current_true_rew = np.mean(rew_list[-avg_w:])
                    s_name = 'learner_{}_steps_{:.1f}_rew_{}_tidx_{}_zpd.tar'.format(
                        str(steps).zfill(7), current_true_rew, tidx, titem)
                    path = os.path.join(params['log']['dir_learner_snapshots'], s_name)
                    agent.save_model_newpath(path)
            matched_list = None
            test_match = False

        # Weights should be synced among this and train/test processes.
        # Called at every multiple of four, so steps = {0,4,8,12,...}.
        agent.train(steps=steps)

        if steps % params["log"]['snapshot_per_step'] < \
                params["train"]['train_freq_per_step'] and \
                steps > params['log']['snapshot_min_step']:
            rew_list = get_true_rew(params["log"]["dir"])
            current_true_rew = np.mean(rew_list[-avg_w:])
            current_clip_rew = np.mean(_play_rewards[-avg_w:])
            agent.save_model(snapshot_number)
            snapshots_summary[snapshot_number] = {
                "clip_rew_life": current_clip_rew,
                "true_rew_epis": current_true_rew,
                "num_finished_epis": len(rew_list),
                "steps": steps,
            }
            snapshot_number += 1
            write_dict(dict_object=snapshots_summary,
                       dir_path=params["log"]["dir_snapshots"],
                       file_name="snapshots_summary")

        # Strictly for debugging. Put in `info_summary` to plot later.
        if steps % params["log"]["debug_per_step"] < \
                params["train"]['train_freq_per_step']:
            true_results = monitor.load_results(params["log"]["dir"])
            logger.info(
                generate_debug_msg(
                    writer=writer, steps=steps, play_rewards=_play_rewards,
                    play_speeds=_play_speeds, test_rewards=_test_rewards,
                    test_speeds=_test_speeds, replay_memory=replay_memory,
                    max_num_steps=params["env"]["max_num_steps"],
                    agent=agent, avg_window=avg_w, true_results=true_results
                )
            )
            time_elapsed_mins = (time.time() - time_start) / 60.0
            info_summary[steps] = {
                'greedy_eps': agent.get_policy_epsilon(steps),
                'agent_lr': agent.get_current_lr(steps),
                'time_elapsed_mins': time_elapsed_mins,
                'frames_per_sec': np.mean(_play_speeds[-avg_w:]),
                'true_avg_rew': np.mean(true_results['r'].tolist()[-avg_w:]),
            }
            if ts.total_num_teachers > 0:
                info_summary[steps]['teacher_blend'] = agent.get_teacher_blend(steps)
                info_summary[steps]['beta_wis'] = agent.get_wis_beta(steps)
            for loss_label, loss_queue in agent.debug_loss.items():
                info_summary[steps][loss_label.replace("_", " ")] = loss_queue.mean()
            write_dict(dict_object=info_summary,
                       dir_path=params["log"]['dir'],
                       file_name="info_summary")

    # --------------------------------------------------------------------------
    # Only way we get here is if we hit env limit, or if we hit `_end`, yet
    # `_end` relies on the process queue being empty. In the train process, from
    # `Tracker` condition, if we hit a sufficiently high reward we break.
    # --------------------------------------------------------------------------
    if _play_process.is_alive():
        play_stop_flag.set()
        while exp_queue.get() is not None:
            pass
        _play_process.join()
    if have_test_proc and _test_process.is_alive():
        test_stop_flag.set()
        while rew_queue.get() is not None:
            pass
        _test_process.join()
    logger.info("Training complete!")


if __name__ == '__main__':
    mp = mp.get_context('spawn')
    # mp.set_start_method('spawn')

    # Read configurations from file
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", type=str, help="ID of the experiment to run")
    parser.add_argument("--exp-name", type=str, default='settings',
                        help='directory from which to load experiment settings')
    parser.add_argument("--profile", action='store_true', help='whether to run the profiler')
    params = parser.parse_args()
    _params = Configurations(params, note="")

    # Include log directory names in the params
    if params.profile:
        main_profiler(_params)
    else:
        main(_params)
