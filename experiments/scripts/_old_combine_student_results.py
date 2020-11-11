"""
Run `python scripts/combine_student_results.py` to coalesce a bunch of student
results together into one figure --- you just need to specify the figure
groupings and names in this file.

This contrasts with `scripts/quick_student.py`, which was designed to quickly
cycle through students and save *one figure per student*. That means we can just
save figures in each student's `figures/` directory.
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

import argparse, csv, math, os, pickle, sys, inspect, json
from os.path import join
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from dqn import common_config as cfg
import utils as U
from collections import defaultdict
np.set_printoptions(linewidth=180, edgeitems=10)

# ------------------------------------------------------------------------------
# matplotlib stuff
# ------------------------------------------------------------------------------
titlesize = 25
xsize = 20
ysize = 20
ticksize = 20
legendsize = 20
student_colors = ['black', 'red', 'blue', 'purple', 'brown']
teacher_color = 'orange'
error_region_alpha = 0.25
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Groups and figure names to inspect. Each (group,figure) pair makes one figure.
# For the groups, put the agent file names (each starts with an underscore).
#
# The code later will test to ensure that all the sub-groupings consist of the
# same teacher, and the plots hopefully should make it clear if I'm copying and
# pasting the wrong files here.
#
# There is a lot of manual work, admittedly. Any ideas to simplify? UPDATE: OK,
# here's how to simplify this. Only list the teacher files that we want. Then
# programmatically go through the student directories and extract all the
# students whose teacher model matches the teacher. THAT'S how we do it. I'll
# only do the manual stuff once, just to show what it SHOULD look like.
#
# It does unfortunately require the files to be present on one computer ...
# which could be a problem if we are short on disk space.
# ------------------------------------------------------------------------------

GROUPS = [
    [
        # Alien 0,4,16,64,b teacher 28728
        [
        '_alien_snapshot_2019-03-28-20-44_s28261',
        '_alien_snapshot_2019-03-27-10-11_s67724', 
        '_alien_snapshot_2019-03-27-10-14_s83656',
        '_alien_snapshot_2019-03-27-10-12_s3093',
        '_alien_snapshot_2019-04-05-21-43_s59884',
        ],
        # Alien 0,4,16,64,b teacher 95763
        [
        '_alien_snapshot_2019-03-28-20-44_s33935',
        '_alien_snapshot_2019-03-27-21-01_s86808',
        '_alien_snapshot_2019-03-27-21-09_s33628',
        '_alien_snapshot_2019-03-27-21-02_s96074',
        '_alien_snapshot_2019-04-06-10-26_s25789',
        ],
    ],
    [
        # Boxing 0,4,16,64 teacher 48495
        [
        '_boxing_snapshot_2019-03-28-20-26_s94059',
        '_boxing_snapshot_2019-03-28-16-21_s35297',
        '_boxing_snapshot_2019-03-28-16-28_s89150',
        '_boxing_snapshot_2019-03-28-16-30_s37427',
        '_boxing_snapshot_2019-04-04-22-20_s3500',
        ],
        # Boxing 0,4,16,64 teacher 7110
        [
        '_boxing_snapshot_2019-03-31-14-44_s54612',
        '_boxing_snapshot_2019-03-31-14-44_s84981',
        '_boxing_snapshot_2019-03-31-20-02_s77864',
        '_boxing_snapshot_2019-03-31-20-02_s69328',
        '_boxing_snapshot_2019-04-05-08-38_s39407',
        ],
        # Boxing 0,4,16,64 teacher 52246
        [
        '_boxing_snapshot_2019-04-01-21-05_s39550',
        '_boxing_snapshot_2019-04-01-15-21_s31253',
        '_boxing_snapshot_2019-04-01-06-48_s7769',
        '_boxing_snapshot_2019-04-01-06-48_s89794',
        '_boxing_snapshot_2019-04-04-22-20_s35580',
        ],
    ],
    [
        # Breakout 0,4,16,64 teacher 20416
        [
        '_breakout_snapshot_2019-03-29-14-16_s56820',
        '_breakout_snapshot_2019-03-29-09-10_s42788',
        '_breakout_snapshot_2019-03-29-09-10_s63306',
        '_breakout_snapshot_2019-03-29-09-10_s18940',
        '_breakout_snapshot_2019-04-05-21-41_s55167',
        ],
        # Breakout 0,4,16,64 teacher 53372
        [
        '_breakout_snapshot_2019-03-29-21-42_s25997',
        '_breakout_snapshot_2019-03-29-21-41_s60651',
        '_breakout_snapshot_2019-03-30-12-59_s11775',
        '_breakout_snapshot_2019-03-30-12-59_s12362',
        '_breakout_snapshot_2019-04-05-14-21_s24385',
        ],
        # Breakout 0,4,16,64 teacher 54456
        [
        '_breakout_snapshot_2019-03-30-21-51_s66862',
        '_breakout_snapshot_2019-03-30-21-51_s65982',
        '_breakout_snapshot_2019-03-31-08-39_s65461',
        '_breakout_snapshot_2019-03-31-08-39_s98721',
        '_breakout_snapshot_2019-04-05-14-20_s32008',
        ],
        # Breakout 0,4,16,64 teacher 37395
        [
        '_breakout_snapshot_2019-04-03-15-14_s55995',
        '_breakout_snapshot_2019-04-02-11-03_s17676',
        '_breakout_snapshot_2019-04-02-18-51_s63885',
        '_breakout_snapshot_2019-04-01-21-21_s13174',
        '_breakout_snapshot_2019-04-05-08-39_s59556',
        ],
    ],
    [
        # Pong 0,4,16,64 teacher 7827
        [
        '_pong_snapshot_2019-03-29-09-22_s78390',
        '_pong_snapshot_2019-03-29-14-20_s29552',
        '_pong_snapshot_2019-03-29-14-24_s45599',
        '_pong_snapshot_2019-03-29-14-24_s14205',
        '_pong_snapshot_2019-04-04-17-00_s81204',
        ],
        # Pong 0,4,16,64 teacher 65253
        [
        '_pong_snapshot_2019-03-29-14-36_s70770',
        '_pong_snapshot_2019-03-29-21-48_s41787',
        '_pong_snapshot_2019-03-30-21-54_s556',
        '_pong_snapshot_2019-03-31-08-40_s45384',
        '_pong_snapshot_2019-04-04-16-59_s81194',
        ],
        # Pong 0,4,16,64 teacher 11963
        [
        '_pong_snapshot_2019-03-31-08-56_s99599',
        '_pong_snapshot_2019-03-31-14-41_s68148',
        '_pong_snapshot_2019-03-31-19-53_s48040',
        '_pong_snapshot_2019-04-01-06-51_s93154',
        '_pong_snapshot_2019-04-04-10-33_s51583',
        ],
        # Pong 0,4,16,64 teacher 72893
        [
        '_pong_snapshot_2019-04-03-21-34_s83941',
        '_pong_snapshot_2019-04-03-21-27_s37409',
        '_pong_snapshot_2019-04-02-18-53_s65027',
        '_pong_snapshot_2019-04-01-21-33_s67814',
        '_pong_snapshot_2019-04-04-10-30_s40075',
        ],
    ],
    [
        # Qbert 0,4,16,64 teacher 65370, lambda 0.1
        [
        '_qbert_snapshot_2019-04-13-17-57_s72241',
        '_qbert_snapshot_2019-04-13-11-34_s33317',
        '_qbert_snapshot_2019-04-13-11-00_s33686',
        '_qbert_snapshot_2019-04-12-09-17_s25821',
        ],
        # Qbert 0,4,16,64 teacher 18837, lambda 0.1
        [
        '_qbert_snapshot_2019-04-14-10-28_s18469',
        '_qbert_snapshot_2019-04-14-10-23_s2935',
        '_qbert_snapshot_2019-04-14-10-23_s13099',
        '_qbert_snapshot_2019-04-14-10-23_s21962',
        ],
    ],
    [
        # Seaquest 0,4,16,64 teacher 56425
        [
        '_seaquest_snapshot_2019-04-02-10-03_s34479', 
        '_seaquest_snapshot_2019-04-02-15-36_s16847',
        '_seaquest_snapshot_2019-04-02-15-34_s25317',
        '_seaquest_snapshot_2019-04-02-12-11_s77556',
        '_seaquest_snapshot_2019-04-06-10-30_s56695',
        ],
        # Seaquest 0,4,16,64 teacher 77017
        [
        '_seaquest_snapshot_2019-04-03-07-23_s93065',
        '_seaquest_snapshot_2019-04-03-07-26_s84840',
        '_seaquest_snapshot_2019-04-03-11-14_s82649',
        '_seaquest_snapshot_2019-04-03-11-14_s66402',
        '_seaquest_snapshot_2019-04-06-10-28_s44243',
        ],
    ],
]

FIGNAMES = [
    'alien_teachers.png',
    'boxing_teachers.png',
    'breakout_teachers.png',
    'pong_teachers.png',
    'qbert_teachers.png',
    'seaquest_teachers.png',
]

fighead = join(cfg.DATA_HEAD, 'figures')
if not os.path.exists(fighead):
    os.makedirs(fighead)
FIGNAMES = [join(fighead, x) for x in FIGNAMES]

assert len(GROUPS) == len(FIGNAMES)
# ------------------------------------------------------------------------------


def plot_one_history(ax, ykey, xkey, data, color, label, w, alpha_fill=0.2,
        plot_std=True, return_data=False):
    """From Allen.
    
    Will add info (usualy total reward) to one of the subplots.  Relies on
    `DataFrame.rolling` which 'provides rolling window calculations', which lets
    us compute mean, std, etc., over a previous window.

    Update: returns the actual data in case we use it later. Use `_smooth`, not
    `_y`, because the latter can be noisy.
    """
    lw = 1
    if 'Teacher' in label:
        lw = 2
    _y = data[ykey]
    assert len(_y) > w
    if xkey == "steps":
        _x = data["cum_steps"] / float(1e6)
    else:
        _x = data.index

    _error  = _y.rolling(window=w, min_periods=1, center=False).std()
    _smooth = _y.rolling(window=w, min_periods=1, center=False).mean()
    ymin = _smooth - _error
    ymax = _smooth + _error
    xdata = _smooth.values.tolist()
    _label = '{}; len/#eps: {}, last {:.1f}'.format(label, len(xdata), xdata[-1])

    ax.plot(_x, _smooth, lw=lw, color=color, label=_label)
    if plot_std:
        ax.fill_between(_x, ymax, ymin, alpha=alpha_fill)
    if return_data:
        return _x, _smooth, _y


def fixed_intervals(xkey, ykey, *, data, w, interval=10000, style=1):
    """Fixed intervals, makes it easier to average reward curves together.

    For example, interval=10k means we look at rewards every 10k steps. Well,
    the past window of rewards, but you get the idea.

    AH, shoot I realized we're looking at the data frame file, but that's
    slightly different from the actual steps the agent took, as I've figured out
    by looking at the steps carefully.

    Also, it's somewhat confusing with pandas, but if you use `_smooth` the way
    we have it, and try to make a list by adding items from `_smooth[idx]`, the
    list will actually look like this:

    [15    184.0
    Name: raw_rew, dtype: float64, 31    195.483871
    Name: raw_rew, dtype: float64, 45    212.444444
    Name: raw_rew, dtype: float64, 60    215.833333
    Name: raw_rew, dtype: float64, 74    214.864865
    Name: raw_rew, dtype: float64]

    so each item in the list is really the episode number and the past 100
    reward, and has the `Name` and `dtype` there. Even moer annoyingly, there is
    a line break after the first two numbers? In any case, I found it easier
    just to call `_smooth[idx].tolist()` which makes a one-item list of the
    *reward*, and then we can simply take the float from that.
    """ 

    #TODO: for now we can use the fixed interval strategy that I have earlier,
    #but we really also want the `info_summary` case ... but that requires me to
    #add a special case for the `root.log` ... sigh. It's going to be slightly
    #off from the 6M steps, but at least it's off uniformly? It will be off but
    #hopefully OK for now. This is style==1.

    _y = data[ykey]
    assert len(_y) > w
    if xkey == "steps":
        _x = data["cum_steps"] / float(1e6)
    else:
        _x = data.index
    _smooth = _y.rolling(window=w, min_periods=1, center=False).mean()

    bxx = []
    byy = []
    for nsteps in range(interval, 6000000+1, interval):
        #https://stackoverflow.com/questions/30112202/
        idx = data.index[(data['cum_steps']-nsteps).abs().argsort()[:1]]
        if xkey == "steps":
            bxx.append(nsteps / float(1e6))
        else:
            bxx.append(nsteps)
        past_w_avg_reward = _smooth[idx].tolist()
        assert len(past_w_avg_reward) == 1
        past_w_avg_reward = past_w_avg_reward[0]
        byy.append(past_w_avg_reward)
    return bxx, byy


def plot_subplot(group, ax, w, style):
    """Plot one subplot, usually means wrt one teacher.

    Returns statistics that we can then use for global information later. To be
    precise, we return a dictionary with keys:
    
        'student_0': (xx, yy, bxx, byy)
            ...
        'student_64': (xx, yy, bxx, byy)
        'teacher': (xx, yy, bxx, byy)

    where the different 'student_k' stuff is for k-steps ahead, and we return
    the (xx,yy) from its specific subplot. However, the problem is those are
    stored each time an episode ends, which means the x-coordinates are not
    consistent among the different trials. That makes it harder to average the
    curves together. Hence, we return bxx, and byy which are 'better' in the
    sense that we get consistent x coordinates. :-)
    """
    result = defaultdict(list)
    exp_paths = []
    titles = []
    summary_trains = []

    for g in group:
        exp_path = join(cfg.SNAPS_STUDENT,g)
        exp_paths.append(exp_path)
        title = U.get_title(exp_path)
        titles.append(title)
        summary_train = U.load_summary_data(exp_path, train=True)
        summary_trains.append(summary_train)

    # --------------------------------------------------------------------------
    # We want to overlay the teacher. Load teacher model, load path, then plot
    # data. Be careful we are allowed to do this 'substitution' to get the
    # expert data file ... and that all the groups here use the same teacher.
    # --------------------------------------------------------------------------
    teacher_models = []
    params = []

    for exp_path in exp_paths:
        with open(join(exp_path,'params.txt'), 'r') as f:
            param = json.load(f)
            params.append(param)
        tms = param['teacher']['models']
        assert len(tms) == 1, "assume len(teachers) = 1, {}".format(len(tms))
        teacher_models.append(tms[0])

    t_model = teacher_models[0]
    for tm in teacher_models:
        assert tm == t_model, "Error w/teachers: {}".format(teacher_models)

    # Plot rewards from training and testing runs. Use `raw_rew`, not `total_rew`.
    for k in range(len(summary_trains)):
        num_ahead = params[k]['teacher']['num_snapshot_ahead']
        label = 'Student, {}'.format(str(num_ahead).zfill(3))
        if '-01' in label:
            label = label.replace('-01','best')
        xx, yy, _ = plot_one_history(
                    ax, w=w,
                    xkey='steps',
                    ykey='raw_rew',
                    data=summary_trains[k],
                    color=student_colors[k],
                    label=label,
                    plot_std=False,
                    return_data=True)
        bxx, byy = fixed_intervals('steps', 'raw_rew',
                data=summary_trains[k], w=w, style=style)
        result['student_'+str(num_ahead).zfill(3)] = (xx, yy, bxx, byy)

    # Pick first exp_path, and replace last portion(s) w/teacher information.
    exp_path = exp_paths[0]
    s_last = os.path.basename(os.path.normpath(exp_path))
    t_last = os.path.basename(os.path.normpath(teacher_models[0]))
    teacher_path = exp_path.replace(s_last, t_last)
    teacher_path = teacher_path.replace('students/', 'teachers/')
    assert os.path.exists(teacher_path), teacher_path
    teacher_title = U.get_title(teacher_path)
    teacher_train = U.load_summary_data(teacher_path, train=True)
    xx, yy, _ = plot_one_history(
            ax, w=w,
            xkey='steps',
            ykey='raw_rew',
            data=teacher_train,
            color=teacher_color,
            label='Teacher',
            plot_std=False,
            return_data=True)
    bxx, byy = fixed_intervals('steps', 'raw_rew',
            data=teacher_train, w=w, style=style)
    ax.set_xlabel("Number of Steps (Millions)", fontsize=xsize)
    ax.set_ylabel("Avg Past {} Rewards".format(w), fontsize=ysize)
    ax.set_title('Teacher: {}'.format(t_model, w), fontsize=titlesize)
    result['teacher'] = (xx, yy, bxx, byy)
    return result


def plot(group, figname, style):
    """Plots stuff from this group.
    """
    game = '...' # later, get this to be better ...
    assert style == 1 or style == 2, style
    w = 100
    ncols = 2
    nrows = int((len(group)+1) / 2) + 1
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
                           sharey=True, figsize=(11*ncols,8*nrows))
    results = defaultdict(list)

    # Each `g` is a list wrt one teacher, ideally. The keys for `results` here
    # will be `student_0`, `student_4`, `teacher`, etc., and values are lists.

    print("Now iterating through students wrt one teacher.")
    for idx,g in enumerate(group):
        print("  group {} ...".format(g))
        row = int(idx / ncols) + 1 # offset by 1
        col = int(idx % ncols)
        res = plot_subplot(g, ax[row,col], w=w, style=style)
        for key in res:
            results[key].append(res[key])

    # Let's average these out and put that info in the first row. Each
    # `results[key]` is a list with length equal to number of trials we ran.
    # The items are the four tuple of (xx, yy, bxx, byy). When we concatenate
    # and form the array, take statistics (mean/std) over axis 0.
    r = 0
    c = 0

    for idx,key in enumerate(sorted(list(results.keys()))):
        xx = np.array([x for (_,_,x,y) in results[key]])
        yy = np.array([y for (_,_,x,y) in results[key]])
        K = xx.shape[0]
        assert xx.shape[0] == len(results[key])
        assert yy.shape[0] == len(results[key])
        assert xx.shape == yy.shape
        x_avg = np.mean(xx, axis=0)
        y_avg = np.mean(yy, axis=0)
        y_std = np.std(yy, axis=0)

        # Put the averaged curves in one plot together.
        if key == 'teacher':
            lw = 2
            label = "{}, avg over {}".format(key, K)
            title = 'Overall, {}'.format(game)
            ax[r,c].plot(x_avg, y_avg, lw=lw, color=teacher_color,
                         label=label)
            ax[r,c+1].plot(x_avg, y_avg, lw=lw, color=teacher_color,
                           label=label)
            ax[r,c+1].fill_between(x_avg, 
                                   y_avg-y_std,
                                   y_avg+y_std,
                                   alpha=error_region_alpha,
                                   facecolor=teacher_color)
            ax[r,c].set_title(title, fontsize=titlesize)
        else:
            lw = 1
            if '-01' in key:
                key = key.replace('-01','best')
            label = "{}, avg over {}".format(key, K)
            title = 'Overall, {}, w/std'.format(game)
            ax[r,c].plot(x_avg, y_avg, lw=lw, color=student_colors[idx],
                         label=label)
            ax[r,c+1].plot(x_avg, y_avg, lw=lw, color=student_colors[idx],
                           label=label)
            ax[r,c+1].fill_between(x_avg, 
                                   y_avg-y_std,
                                   y_avg+y_std,
                                   alpha=error_region_alpha,
                                   facecolor=student_colors[idx])
            ax[r,c].set_title(title, fontsize=titlesize)

    # Bells and whistles
    for row in range(nrows):
        for col in range(ncols):
            leg = ax[row,col].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[row,col].tick_params(axis='x', labelsize=ticksize)
            ax[row,col].tick_params(axis='y', labelsize=ticksize)
            # I think it's better to share axes in the x direction to be
            # consistent with steps, but doing so removes the axis ticks. This
            # reverts it so we get the ticks on all the axis.
            ax[row,col].xaxis.set_tick_params(which='both', labelbottom=True)
            ax[row,col].set_xlabel("Number of Steps (Millions)", fontsize=xsize)
            ax[row,col].set_ylabel("Avg Past {} Rewards".format(w), fontsize=ysize)

    # Finally, save!!
    plt.tight_layout()
    plt.savefig(figname)
    print("Just saved:\n\t{}\n".format(figname))


if __name__ == "__main__":
    # See the top of this script for the groups to use.  The only argument I'll
    # add is for how we want to plot the results and to format the x-axis values
    # to be consistent.

    STYLE = 1  # Just map to closest value in 0 to slightly over 6M steps.
    #STYLE = 2  # Use fixed 10k interval, reading from root.log for older files

    for (group,figname) in zip(GROUPS,FIGNAMES):
        plot(group, figname, STYLE)
