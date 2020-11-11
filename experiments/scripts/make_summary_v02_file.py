"""Given an agent file, parse the log file to make a similar summary dictionary.

Example:

    python scripts/make_summary_v02_file.py  \
            --agent _pong_standard_2019-03-20-10-29_s7827

This will populate a dictionary (in text form) `training_summary_true_v02.txt`
in the agent's `episodes/` directory.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-darkgrid')
sns.set_style("darkgrid")
from matplotlib import pyplot as plt
from matplotlib import offsetbox
from matplotlib.ticker import FuncFormatter
import argparse
import csv
import math
import os
import pickle
import sys
import inspect
import json
from os.path import join
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from dqn import common_config as cfg
import utils as U
from dqn.utils.io import write_dict

# -------------------------
# matplotlib stuff
titlesize = 33
xsize = 30
ysize = 30
ticksize = 25
legendsize = 25
# -------------------------


def make_summary(agent_path):
    # If train_v02 exists, delete it and re-run.
    episodes  = join(agent_path, 'episodes')
    train_log = join(agent_path, 'train.log')  # NOT root.log !!
    train_v01 = join(episodes, 'training_summary_true.txt')
    train_v02 = join(episodes, 'training_summary_true_v02.txt')
    assert os.path.exists(train_log), train_log
    assert os.path.exists(train_v01), train_v01
    assert not os.path.exists(train_v02), train_v02

    # Parse data from old summary file. This just makes exact same info.
    summary_true_v02 = {}
    with open(train_v01, 'r') as fh:
        data = json.load(fh)
    ep_idx = 1
    while str(ep_idx) in data:
        summary_true_v02[int(ep_idx)] = data[str(ep_idx)]
        ep_idx += 1
    num_true_episodes = len(data.keys())
    assert ep_idx == num_true_episodes + 1

    # Now let's add new info with help of train.log. It does assume a fixed way
    # of logging train.log but whatever ...
    with open(train_log) as fh:
        log = fh.readlines() # :(
    assert len(log) > 100, len(log) # sanity

    print("we have {} true episodes".format(num_true_episodes))
    print("loaded log file of length {}".format(len(log)))
    life_start = 1

    # Go through, detect 'Episode', then use the line before that for last life
    # that was in that episode.
    for lidx,line in enumerate(log):

        if 'Episode' in line and 'done' in line and 'raw reward' in line:
            print(log[lidx-1].rstrip())
            print(line.rstrip())
            line_life = log[lidx-1].rstrip().split()
            line_epis = log[lidx].rstrip().split()

            for idx,item in enumerate(line_life):
                if item == 'DEBUG':
                    assert line_life[idx+1] == 'Life', line_life
                    life_end = int(line_life[idx+2])
            for idx,item in enumerate(line_epis):
                if item == 'DEBUG':
                    assert line_epis[idx+1] == 'Episode', line_epis
                    epis_num = int(line_epis[idx+2])

            print(life_end, epis_num, '\n')
            summary_true_v02[epis_num]['life_idx_begin'] = life_start
            summary_true_v02[epis_num]['life_idx_end'] = life_end
            summary_true_v02[epis_num]['life_num'] = life_end - life_start + 1
            life_start = life_end + 1

    write_dict(dict_object=summary_true_v02,
               dir_path=episodes,
               file_name='training_summary_true_v02')

    # Sanity check for games that have constant lives per episode.
    # AHA! This caught that SeaQuest has some non-4 life games!!
    for ep_idx in range(1, num_true_episodes+1):
        if 'pong' in agent_path or 'boxing' in agent_path:
            assert summary_true_v02[ep_idx]['life_num'] == 1, \
                summary_true_v02[ep_idx]
        elif 'alien' in agent_path:
            assert summary_true_v02[ep_idx]['life_num'] == 3, \
                summary_true_v02[ep_idx]
        elif 'qbert' in agent_path:
            assert summary_true_v02[ep_idx]['life_num'] == 4, \
                summary_true_v02[ep_idx]
        elif 'breakout' in agent_path:
            assert summary_true_v02[ep_idx]['life_num'] == 5, \
                summary_true_v02[ep_idx]


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--agent', type=str)
    args = pp.parse_args()
    assert args.agent is not None
    AGENT = join(cfg.SNAPS_TEACHER, args.agent)
    make_summary(AGENT)
