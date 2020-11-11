"""Designed for fast inspection of standard DDQN results.

Run these to cycle through all the DDQN results so that we can plot them:

    `python experiments/scripts/quick_dqn.py`

The figures will be created within each agent's training log directory. That is,
it doesn't currently aggregate them all into one plot.

NOTE: `import matplotlib` and `matplotlib.use('Agg')` MUST be the very first two
lines in the code. Otherwise we get Tkinter errors. Just do it.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-darkgrid')
from matplotlib import pyplot as plt
from matplotlib import offsetbox
from matplotlib.ticker import FuncFormatter
sns.set_style("darkgrid")
import argparse, csv, math, os, pickle, sys, inspect
from os.path import join
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from dqn import common_config as cfg
import utils as U

# -------------------------
# matplotlib stuff
titlesize = 33
xsize = 30
ysize = 30
ticksize = 25
legendsize = 25
# -------------------------

# Ignore these since I may have had outdated keys, etc.
STUFF_TO_SKIP = [
]


def plot_one_history(ax, ykey, xkey, data, color, label, window=100,
        alpha_fill=0.2):
    """From Allen. Will add info (usualy total reward) to one of the subplots.
    Relies on `DataFrame.rolling` which 'provides rolling window calculations',
    which lets us compute mean, std, etc., over a previous window.
    """
    _y = data[ykey]
    assert len(_y) > window
    if xkey == "steps":
        _x = data["cum_steps"] / float(1e6)
    else:
        _x = data.index
    _error  = _y.rolling(window=window, min_periods=1, center=False).std()
    _smooth = _y.rolling(window=window, min_periods=1, center=False).mean()
    ymin = _smooth - _error
    ymax = _smooth + _error
    xdata = _smooth.values.tolist()
    _label = '{}; len {}, last {:.1f}'.format(label, len(xdata), xdata[-1])
    ax.plot(_x, _smooth, color=color, label=_label)
    ax.fill_between(_x, ymax, ymin, alpha=alpha_fill)


def plot(exp_path, nrows=2, ncols=2):
    figdir = join(exp_path,'figures')
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    title = U.get_title(exp_path)
    summary_train = U.load_summary_data(exp_path, train=True)
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(11*ncols,8*nrows))

    # Plot rewards from training and testing runs. Use `raw_rew`, not `total_rew`.
    plot_one_history(ax[0,0], xkey='steps', ykey='raw_rew',
            data=summary_train, color='red', label='Train')
    ax[0,0].set_xlabel("Training Steps (Millions)", fontsize=xsize)
    ax[0,0].set_ylabel("Rewards", fontsize=ysize)
    ax[0,0].set_title(title, fontsize=titlesize)

    # Also the snapshots we saved ...
    snapshots = U.load_snapshots_data(exp_path)
    srews  = snapshots['true_rew_epis'].values.tolist()
    ssteps = snapshots['steps'] / 1e6
    ax[0,1].set_xlabel("Training Steps (Millions)", fontsize=xsize)
    ax[0,1].set_ylabel("Rewards", fontsize=ysize)
    ax[0,1].set_title(
        "{} Saved Snapshots; max {:.1f}".format(len(snapshots), np.max(srews)),
        fontsize=titlesize)
    ax[0,1].plot(ssteps, srews, marker='x',
        label="Snapshot Rew, last {:.1f}".format(srews[-1]))

    # Other interesting info that I saved
    i_steps, info = U.load_info_data(exp_path)
    i_steps = np.array(i_steps) / 1e6
    lr   = info['agent_lr'].values
    eps  = info['greedy_eps'].values
    mins = info['time_elapsed_mins'].values
    ax[1,0].set_xlabel("Training Steps (Millions)", fontsize=xsize)
    ax[1,1].set_xlabel("Training Steps (Millions)", fontsize=xsize)
    #ax[1,0].set_ylabel("Agent Learning Rate", fontsize=ysize)
    ax[1,1].set_ylabel("Agent Greedy Epsilon", fontsize=ysize)
    ax[1,0].set_title("Learning Rate; Time {:.1f} Hours".format(mins[-1]/60.0),
                      fontsize=titlesize)
    ax[1,1].set_title("Greedy Epsilon; Time {:.1f} Mins".format(mins[-1]),
                      fontsize=titlesize)
    ax[1,0].plot(i_steps, lr, marker='x',
            label="Learning Rate, last {:.5f}".format(lr[-1]))
    ax[1,1].plot(i_steps, eps, marker='x',
            label="Greedy Epsilon, last {:.2f}".format(eps[-1]))

    # Bells and whistles
    for row in range(nrows):
        for col in range(ncols):
            leg = ax[row,col].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[row,col].tick_params(axis='x', labelsize=ticksize)
            ax[row,col].tick_params(axis='y', labelsize=ticksize)

    # Finally, save!!
    plt.tight_layout()
    lastname = 'fig_train_results_{}.png'.format(title)
    figname = join(figdir,lastname)
    plt.savefig(figname)
    print("Just saved:\n\t{}\n".format(figname))


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # ADJUST PATH(S)!!! Convention is to use `final` for final runs, `settings`
    # for stuff in progress, though eventually we'll move to hard disk storage.
    # --------------------------------------------------------------------------
    EXP_PATH = cfg.SNAPS_TEACHER
    MONTH_CUTOFF = 5

    def criteria(x):
        if 'settings' in x:
            return False
        if '_old' in x:
            return False
        if 'tar.gz' in x:
            return False
        assert x[0] == '_', x
        ss = x.split('_')
        if 'fast' in x:
            assert len(ss) == 6, x
            date = (ss[4]).split('-')
            year, month, day = int(date[0]), int(date[1]), int(date[2])
        else:
            assert len(ss) == 5, x
            date = (ss[3]).split('-')
            year, month, day = int(date[0]), int(date[1]), int(date[2])
        assert year == 2019, year
        if month >= MONTH_CUTOFF:
            return True
        else:
            return False

    dirs = sorted( [join(EXP_PATH,x) for x in os.listdir(EXP_PATH) if criteria(x)] )
    print("Currently plotting with these models, one trained agent per file:")
    for dd in dirs:
        last_part = os.path.basename(os.path.normpath(dd))
        if last_part in STUFF_TO_SKIP or 'tar.gz' in last_part:
            print("  skipping {}".format(last_part))
            continue
        print("  {}".format(dd))
        plot(dd)
