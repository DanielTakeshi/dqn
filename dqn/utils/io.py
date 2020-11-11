import os
import shutil
import json
import pickle
import pandas as pd
import torch


def dump_pickle(_object, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(_object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b


def create_sub_dir(params):
    """Add different subdirectories here.

    New keys are introduced in the params[log], and can use later in code.
    Applies for whatever is the current log dir, for students or teachers, it
    will apply only to its own log dir. Thus, teacher agents actually have
    `learner_snapshots` as subdirs, though they don't put anything there.
    """
    for subdirectory in ["snapshots", "episodes", "learner_snapshots"]:
        _param_name = "dir_{0}".format(subdirectory)
        params["log"][_param_name] = os.path.join(params["log"]["dir"],
                                                  subdirectory)
        os.makedirs(params["log"][_param_name])


def create_output_dir(params):
    """From `Configurations` class to create output directory.
    """
    assert not os.path.exists(params["log"]["dir"]), \
        "Error, {} exists (unlucky random seed?)".format(params['log']['dir'])
    os.makedirs(params["log"]["dir"])
    create_sub_dir(params)


def read_json(dir_path, file_name):
    if ".txt" in file_name:
        file_name = file_name
    else:
        file_name = "{0}.json".format(file_name)
    with open(os.path.join(dir_path, file_name), 'r') as f:
        data = json.load(f)
    return data


def write_dict(dict_object, dir_path, file_name):
    with open(os.path.join(dir_path, "{0}.txt".format(file_name)), 'w') as f:
        json.dump(dict_object, f, sort_keys=True, indent=4)
        f.write('\n')


def save_trajectory(dir_episodes, episode, trajectory, flag="play"):
    """Does the memory-consuming process of saving the trajectory.

    Might consider adding a parameter for this. It pickle-dumps `trajectory`
    which must contain all the frames. Yeowch! There is already a parameter for
    the _colored_ output, which we can always set to false for now.

    Parameters
    ----------
    dir_episodes: self.log_params["dir_episodes"]
    episode: integer, representing episode index (edit: lifespan)
    trajectory: a `dqn.replay.episode.Episode` object
    flag: either 'train' or 'test', I think, not 'play'
    """
    lifespan = "episode_{}_{}.pkl".format(flag, str(episode).zfill(7))
    with open(os.path.join(dir_episodes, lifespan), 'wb') as _trajectory_file:
        pickle.dump(trajectory, _trajectory_file)


def read_trajectory(dir_episodes, episode, flag="train"):
    lifespan = "episode_{}_{}.pkl".format(flag, str(episode).zfill(7))
    _episode_path = os.path.join(dir_episodes, lifespan)
    if os.path.exists(_episode_path):
        with open(_episode_path, 'rb') as f:
            _trajectory = pickle.load(f)
        return _trajectory
    else:
        return None


def read_snapshot(model_dir, number_0_idx, gpu, gpu_id):
    """Read snapshot from zero indexed number.
    """
    number = number_0_idx + 1
    _snapshot_loc = os.path.join(model_dir, "snapshots",
        "snapshot_{0}.pth.tar".format(str(number).zfill(4)))
    assert os.path.exists(_snapshot_loc), _snapshot_loc
    assert isinstance(gpu, bool)
    if gpu:
        _snapshot = torch.load(
            _snapshot_loc,
            map_location=lambda storage, loc: storage.cuda(gpu_id))
    else:
        _snapshot = torch.load(_snapshot_loc, map_location="cpu")
    return _snapshot


def read_episode(model_dir, number_0_idx):
    """Read a LIFESPAN (not episode despite name) from zero indexed number.
    """
    number = number_0_idx + 1
    lifespan = "episode_train_{}.pkl".format(str(number).zfill(7))
    _episode_loc = os.path.join(model_dir, "episodes", lifespan)
    assert os.path.exists(_episode_loc), _episode_loc
    return read_pickle(filename=_episode_loc)
