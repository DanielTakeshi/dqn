"""Utility files."""
import os, sys
from os.path import join
import numpy as np
import pandas as pd


# Helps with plotting per-game stuff.
GAMES = ['Alien', 'Beamrider', 'Boxing', 'Breakout', 'Pong', 'Qbert',
        'Robotank', 'Seaquest', 'SpaceInvaders']

# For a 'fat' plot with all 9 games ((0,0) reserved for legend).
G_INDS_FAT = [(0,1), (0,2), (0,3), (0,4), (1,0), (1,1), (1,2), (1,3), (1,4)]

# For a 'tall' plot wiht all 9 games ((0,0) reserved for legend).
G_INDS_TALL = [(1,0), (2,0), (3,0), (4,0), (0,1), (1,1), (2,1), (3,1), (4,1)]

# For a 'square' plot wiht all 9 games (but no space for a legend).
G_INDS_SQUARE = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]

# For 4.0 experiments, 1M/50k buffers, 75/25 minibatch.
G_OLAP_01 = ['0.3', '0.4', '0.1', '0.3', '0.3', '0.1', '0.1', '0.1', '0.1']

# For 4.2 experiments, ablation with 250k/250k buffers, 75/25 minibatch.
G_OLAP_02 = ['0.3', '0.4', '0.1', '0.3', '0.3', '0.1', '0.1', '0.2', '0.1']

# For 4.2 experiments, ablation with 250k/250k buffers, 50/50 minibatch.
G_OLAP_03 = ['0.3', '0.4', '0.1', '0.3', '0.3', '0.1', '0.1', '0.2', '0.1']

# For 4.2 experiments, ablation with 250k/250k buffers, 50/50 minibatch, 2x more updates.
# Haven't done the overlaps for these yet -- because I think we're moving away from it.
G_OLAP_04 = ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']

# 250k/250k buffers, 25/75 minibatch, 2x more updates.
# Haven't done the overlaps for these.
G_OLAP_05 = ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']


# Ignore these since I may have had outdated keys, etc.
STUFF_TO_SKIP = [
    '_breakout_snapshot_2019-07-30-09-29_s62729',       # testing overfitting at end
    '_breakout_snapshot_2019-07-30-13-49_s19173',       # testing overfitting at end
    '_qbert_snapshot_2019-07-27-16-19_s5584',           # did decay to 0.05 (best ahead)
    '_qbert_snapshot_2019-07-28-11-25_s5962',           # did decay to 0.05 (best ahead)
    '_seaquest_snapshot_2019-07-30-09-48_s88108',       # testing overfitting at end
    '_seaquest_snapshot_2019-07-26-07-37_s29408',       # did decay to 0 (2 ahead)
    '_seaquest_snapshot_2019-07-26-13-31_s44946',       # did decay to 0 (2 ahead)
    '_seaquest_snapshot_2019-07-19-14-32_s84293',       # did decay to 0 (train_net 0.3)
    '_pong_snapshot_2019-08-06-08-47_s38729',           # oops, 64 ahead w/condense 5
    '_pong_snapshot_2019-08-06-08-47_s44561',           # oops, 64 ahead w/condense 5
    '_pong_snapshot_2019-08-06-14-26_s28790',           # oops, 64 ahead w/condense 5
    '_pong_snapshot_2019-08-06-14-26_s99422',           # oops, 64 ahead w/condense 5
    '_spaceinvaders_snapshot_2019-08-07-15-13_s1265',   # before doing decay to 0.05
    '_spaceinvaders_snapshot_2019-08-07-21-27_s68387',  # before doing decay to 0.05
    '_alien_snapshot_2019-08-15-10-46_s26985',          # not sure, terminated very early
    '_beamrider_snapshot_2019-08-28-19-40_s6891',       # I tested lambda 0.1; it was bad
    '_robotank_snapshot_2019-08-28-19-40_s90653',       # I tested lambda 0.1; it was bad
    '_robotank_snapshot_2019-09-03-09-48_s75976',       # Oops, lambda 0.1 by mistake.
]


def _criteria_for_experiments(x, args):
    """We might want this more restrictive than in the quick_student code.

    AH unfortunately this will fail if we try to add more results to earlier
    tiers ... argh, leaving this as a hack and we'll figure out how to fix this
    later. We can just do this as a first round filter, and then LATER filter
    by checking for certain conditions in the saved configuration file.
    """
    if args.exp == 1:
        # For original '4.0' experiments, 1M/50k, 75/25 mb, 4:1 step:update.
        MONTH_BEGIN, DAY_BEGIN = 7, 18
        MONTH_END, DAY_END = 8, 6
    elif args.exp == 2:
        # 4.2 experiments, 250k/250k, 75/25 mb, 4:1 step:update.
        MONTH_BEGIN, DAY_BEGIN = 8, 6
        MONTH_END, DAY_END = 8, 15
    elif args.exp == 3:
        # 4.2 experiments, 250k/250k, 50/50 mb, 4:1 step:update.
        MONTH_BEGIN, DAY_BEGIN = 8, 11
        MONTH_END, DAY_END = 9, 10
    elif args.exp == 4:
        # 4.2 experiments, 250k/250k, 50/50 mb, 2:1 step:update.
        MONTH_BEGIN, DAY_BEGIN = 8, 15
        MONTH_END, DAY_END = 9, 10
    elif args.exp == 5:
        # 4.2 experiments, 250k/250k, 25/75 mb, 2:1 step:update.
        MONTH_BEGIN, DAY_BEGIN = 8, 28
        MONTH_END, DAY_END = 9, 10
    else:
        raise ValueError(ETYPE)

    if 'settings' in x or '__old' in x:
        return False
    assert x[0] == '_', x
    ss = x.split('_')
    if 'fast' in x:
        assert len(ss) == 6, x
        date = (ss[4]).split('-')
    else:
        assert len(ss) == 5, x
        date = (ss[3]).split('-')
    year, month, day = int(date[0]), int(date[1]), int(date[2])
    assert year == 2019, year

    begin = (month > MONTH_BEGIN) or (month == MONTH_BEGIN and day >= DAY_BEGIN)
    end   = (month < MONTH_END)   or (month == MONTH_END   and day <= DAY_END)
    return begin and end


def _criteria_for_experiments_throughput(x, args):
    """We have to deal with two sets of experiments here.

    args.exp == 3:
    # 4.2 experiments, 250k/250k, 50/50 mb, 4:1 step:update.
    MONTH_BEGIN, DAY_BEGIN = 8, 11
    MONTH_END, DAY_END = 9, 10

    args.exp == 4:
    # 4.2 experiments, 250k/250k, 50/50 mb, 2:1 step:update.
    MONTH_BEGIN, DAY_BEGIN = 8, 15
    MONTH_END, DAY_END = 9, 10

    We'll do additional filtering outside of this method.
    """
    MONTH_BEGIN, DAY_BEGIN = 8, 11
    MONTH_END, DAY_END = 9, 10

    if 'settings' in x or '__old' in x:
        return False
    assert x[0] == '_', x
    ss = x.split('_')
    if 'fast' in x:
        assert len(ss) == 6, x
        date = (ss[4]).split('-')
    else:
        assert len(ss) == 5, x
        date = (ss[3]).split('-')
    year, month, day = int(date[0]), int(date[1]), int(date[2])
    assert year == 2019, year

    begin = (month > MONTH_BEGIN) or (month == MONTH_BEGIN and day >= DAY_BEGIN)
    end   = (month < MONTH_END)   or (month == MONTH_END   and day <= DAY_END)
    return begin and end


def smoothed(x, w):
    """Smooth x by averaging over windows of w, assuming sufficient length.
    """
    if len(x) <= w:
        return x
    smooth = []
    for i in range(1, w):
        smooth.append( np.mean(x[0:i]) )
    for i in range(w, len(x)+1):
        smooth.append( np.mean(x[i-w:i]) )
    assert len(x) == len(smooth), "lengths: {}, {}".format(len(x), len(smooth))
    return np.array(smooth)


def remove_date(string):
    """Remove date from our strings.
    Examples:

    pong_standard_fast_2018-12-06-17-15_s36550 -> pong_standard_fast_s36550
    0    1        2    3                4
    seaquest_standard_2019-01-14-21-14_s86594 -> seaquest_standard_s86594
    0        1        2                3

    We call `get_title` before this, so we remove the leading underscores.
    A bunch of heuristics here ... we should get a better solution someday.
    """
    ss = string.split('_')
    if 'pong' in string and 'fast' in string:
        assert len(ss) == 5, ss
        result = "_".join( [ss[0],ss[1],ss[2],ss[4]] )
    else:
        assert len(ss) == 4, ss
        result = "_".join( [ss[0],ss[1],ss[3]] )
    return result


def get_game_name(s_last):
    """Assumes s_list is the directory of the student agent.
    """
    assert s_last[0] == '_', s_last
    name = (s_last[1:]).split('_')
    return name[0]


def get_title(exp_path):
    """Get title from a path, we might change convention so watch out.

    AH, so `exp_path` is designed to be so that we have `[path...]/[agent]`
    where [agent] starts with an underscore. That's why we use a [-1]. But if
    this ends with a slash that's a problem, which is why I remove it.
    """
    if exp_path[-1] == '/':
        exp_path = exp_path[:-1]
    result = (exp_path.split('/'))[-1]
    result = result[1:] # remove leading underscore
    result = remove_date(result)
    return result


def load_snapshots_data(exp_path):
    """Load snapshot data from Allen's code.
    """
    snp = pd.read_json(
        join(exp_path, "snapshots/snapshots_summary.txt"), orient="index")
    snp.index = snp.index.map(int)
    snp.sort_index(inplace=True)
    return snp


def load_info_data(exp_path):
    """Load data that I made. :-) I extract `sorted_steps` here since I didn't
    actually save it, whoops.
    """
    snp = pd.read_json(join(exp_path, "info_summary.txt"), orient="index")
    snp.index = snp.index.map(int)
    sorted_steps = sorted(snp.index.values.tolist())
    snp.sort_index(inplace=True)
    return sorted_steps, snp


def load_summary_data(exp_path, train=True):
    """From Allen. Load summary text files from dqn logs.
    Also adds a 'cum_steps' key, since we saved steps per episode.

    TO BE CLEAR: this is the 'episodes' summary, NOT the 'snapshots' summary.

    NOTE: modified this to use true rewards, not the clipped ones.  ALERT ...
    right now we start episodes from index 0, whereas earlier Allen had it at
    index 1 ... careful if this affects any indexing. I fixed this after
    noticing it but there are 7 Pong trials and 2 Seaquest ones that use the
    earlier indexing.

    May 2019: updating to v02 so that we force us to use the more up-to-date
    summary file.
    """
    _summary_flag = "training" if train else "testing"
    path = join(exp_path, "episodes", _summary_flag + "_summary_true_v02.txt")
    assert os.path.exists(path), path
    summary = pd.read_json(path, orient="index")
    summary.index = summary.index.map(int)
    summary.sort_index(inplace=True)
    summary["cum_steps"] = summary["steps"].cumsum()
    return summary


def criteria_for_quick_student(x):
    """Used for filtering items for our scripts.

    Putting this here to avoid having to copy this method in scripts.
    According to https://github.com/CannyLab/dqn/issues/63, it seems like the
    earliest dates I see are from 07/18.

    Also, for anything post-NeurIPS 2019 workshop, we want to be looking at
    October 21 and onwards.
    """
    MONTH_CUTOFF = 10
    DAY_CUTOFF = 21
    if 'settings' in x or '__old' in x:
        return False
    assert x[0] == '_', x
    ss = x.split('_')
    if 'fast' in x:
        assert len(ss) == 6, x
        date = (ss[4]).split('-')
    else:
        assert len(ss) == 5, x
        date = (ss[3]).split('-')
    year, month, day = int(date[0]), int(date[1]), int(date[2])
    assert year == 2019, year
    return (month > MONTH_CUTOFF) or (month == MONTH_CUTOFF and day >= DAY_CUTOFF)
