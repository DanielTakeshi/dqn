"""Plot for steps per second, and maybe other things.

Just run `python scripts/steps_per_sec.py`.

NOTE: `import matplotlib` and `matplotlib.use('Agg')` MUST be the very first two
lines in the code. Otherwise we get Tkinter errors. Just do it.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import offsetbox
from matplotlib.ticker import FuncFormatter
import argparse, csv, math, os, pickle, sys, inspect, json
from os.path import join
import numpy as np
import pandas as pd
from dqn import common_config as cfg
import utils as U
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'figure.max_open_warning': 100})
sns.set_style("darkgrid")
np.set_printoptions(suppress=True, linewidth=180, precision=3)
from collections import defaultdict

# ------------------------------------------------------------------------------
# matplotlib stuff
# ------------------------------------------------------------------------------
titlesize = 48
xsize = 40
ysize = 40
ticksize = 40
legendsize = 48
colors = ['black', 'red', 'blue', 'purple', 'brown', 'orange']
error_region_alpha = 0.25
bwidth = 0.3
# ------------------------------------------------------------------------------
CONST = 1e6

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
]


# ADJUST
CASE = 1

if CASE == 1:
    # Use these for when I was running 1 trial on 1 GPU.  Uses the 50/50 mb,
    # 250k/250k, with train_freq_update of 4, so not directly comparable?
    MONTH_BEGIN, DAY_BEGIN = 8, 11
    MONTH_END,   DAY_END   = 8, 13
elif CASE == 2:
    # Use these for when I was running 1 trial on 1 GPU.  Uses the 50/50 mb,
    # 250k/250k, with train_freq_update of 4, so not directly comparable?
    MONTH_BEGIN, DAY_BEGIN = 8, 15
    MONTH_END,   DAY_END   = 8, 18
elif CASE == 3:
    # Use these for when I was running on average 2 trials per GPU.
    # Uses the 50/50 mb, 250k/250k, with train_freq_update of 2.
    MONTH_BEGIN, DAY_BEGIN = 8, 21
    MONTH_END,   DAY_END   = 8, 21
else:
    raise ValueError(CASE)


def scale_steps(x):
    x = np.array(x) / CONST
    return x


def plot_one_history(ax, xkey, ykey, data, color, label, w, alpha_fill=0.2):
    """From Allen. Will add info (usualy total reward) to one of the subplots.

    Relies on `DataFrame.rolling` which 'provides rolling window calculations',
    which lets us compute mean, std, etc., over a previous window. Must use the
    same scale (1e6) as the `scale_steps` method.

    Update I: returns the actual data in case we use it later, (_x,_smooth)
    which are for the steps and the smoothed game sore, respectively. Also, from
    two GitHub issues, I think we can add another row to the legend label which
    shows the actual things I want to see, the average reward over all episodes,
    and the average over the last 100. As a sanity check, that last 100 should
    match the 100-smoothed last value.

    Update II: adding another row/statistic so that we only deal with the stuff
    after the first 50k steps. I'm assuming that 50k steps is the exploration
    period, so we start averaging data['raw_rew'] when data['cum_steps'] exceeds
    50k. Note unfortunately that data['cum_steps'] is NOT the same as the actual
    steps iteration, it overestimates a bit (it's a known issue, see the issues
    tracker on GitHub). It's _close_ so that's why the plots look like we're
    using 6M steps exactly.
    """
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
    _label = '{}; len/#eps: {},\n(sm) avg/max/last: {:.1f}, {:.1f}, {:.1f}'.format(
            label, len(xdata), np.mean(xdata), np.max(xdata), xdata[-1])

    # Ignore the exploration period.
    target = -1
    explore_period = 50000
    yy = np.array(data['cum_steps'])
    for idx,i1 in enumerate(yy):
        if i1 > explore_period:
            target = idx
            break
    yy = _y[target:]  # overwrite yy, so now it has rewards
    assert len(yy) < len(_y), '{} vs {}'.format(len(yy), len(_y))

    # https://github.com/CannyLab/dqn/issues/50
    # https://github.com/CannyLab/dqn/issues/53
    _label += "\n(raw) avg1,2/last100: {:.1f}, {:.1f}, {:.1f}".format(
            np.mean(_y), np.mean(yy), np.mean(_y[-100:]))

    ax.plot(_x, _smooth, lw=3, color=color, label=_label)
    ax.fill_between(_x, ymax, ymin, alpha=alpha_fill)
    return _x, _smooth


def get_info(exp_path, w=100):
    """Gather information, in a similar manner as scripts/quick_student.py.

    Be sure that teacher paths exist, and are untarred. Also, `exp_path` is the
    full path to the learner agent. The goal is to get the following information:

    - Average reward
        - The average over training (and ignoring first 50k) and the final
          window of 100, for reporting.
        - Be sure to use `raw_rew` for the y-key, not `total_rew`.
    - Which overlap type
        - And the corresponding overlap parameter
    """
    title = U.get_title(exp_path)
    summary_train = U.load_summary_data(exp_path, train=True)

    # Get the teacher info. Load teacher model, load path, then plot data. Be
    # careful we are allowed to do this 'substitution' to get the expert data.
    with open(join(exp_path,'params.txt'), 'r') as f:
        params = json.load(f)
    teacher_models = params['teacher']['models']
    assert len(teacher_models) == 1, \
            "assume len(teacher_models) = 1, {}".format(len(teacher_models))
    s_last = os.path.basename(os.path.normpath(exp_path))
    t_last = os.path.basename(os.path.normpath(teacher_models[0]))
    teacher_path = exp_path.replace(s_last, t_last)
    teacher_path = teacher_path.replace('students/', 'teachers/')
    teacher_title = U.get_title(teacher_path)
    teacher_train = U.load_summary_data(teacher_path, train=True)

    # Other interesting info that I saved from student's learning progress.
    # If an error results, check that the run didn't terminate early, etc.
    i_steps, info = U.load_info_data(exp_path)
    i_steps = scale_steps(i_steps)
    # Sanity check. Breakout may be longer as training waits until episodes finish.
    assert 6.00 <= np.max(i_steps) <= 6.10, i_steps
    t_lambda    = params['teacher']['supervise_loss']['lambda']
    t_condense  = params['teacher']['condense_freq']
    t_overlap_m = params['teacher']['overlap']['match_method']
    if t_overlap_m == 'train_net':
        t_overlap_p = params['teacher']['overlap']['overlap_target']
    elif t_overlap_m == 'fixed_steps':
        t_overlap_p = str(params['teacher']['num_snapshot_ahead']).zfill(2)
        assert t_condense == 5, t_condense
    else:
        raise ValueError(t_overlap_m)

    # For now
    if 'beamrider' in s_last.lower() or 'pong' in s_last.lower() or \
            'robotank' in s_last.lower():
        assert t_lambda == 0.01, '{}, {}'.format(t_lambda, s_last)
    else:
        assert t_lambda == 0.1, '{}, {}'.format(t_lambda, s_last)

    # the 'frames_per_sec' is really 'environment steps per second'.
    result = {
        'game_name':        U.get_game_name(s_last),
        'frames_per_sec':   info['frames_per_sec'],
        'overlap_param':    t_overlap_p,
        'match_method':     t_overlap_m,
        'supervise_lambda': t_lambda,
        'mb_start':         params['teacher']['blend']['start'],
        'mb_end':           params['teacher']['blend']['end'],
    }
    return result


def report_combined_stats(stats):
    """Report combined stats, ideally for a table or plot in a paper.
    """
    nrows = 2
    ncols = 5
    w = 100
    # Increase factor to `nrows` to make plot 'taller'.
    # Note: I actually enforce an axes limit later.
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=False, sharey=True,
                           figsize=(11*ncols,10*nrows))

    # Data for plots later
    all_game_stats = []  # list of dicts, one per game
    t_stats = defaultdict(list)

    # Go through and print text I can copy/paste, while adding to the plot.
    game_idx = 0
    print('\n\n\t\tNEW GAME: {}'.format(U.GAMES[game_idx]))
    game_info = {}

    for key in sorted(stats.keys()):
        game = U.GAMES[game_idx]
        if game.lower() not in key:
            game_idx += 1
            game = U.GAMES[game_idx]
            print('\n\n\t\tNEW GAME: {}'.format(game))
            # Add the previously accumulated states to the game_stats.
            all_game_stats.append(game_info)
            game_info = {}
        num_trials = len(stats[key])
        print('\n{}   len(stats[key]): {}'.format(key, num_trials))

        steps_per_sec = [x['frames_per_sec'] for x in stats[key]]
        minlen = np.float('inf')
        for ss in steps_per_sec:
            print(' ', ss.shape)
            minlen = min(minlen, (ss.shape)[0])
        for idx in range(len(steps_per_sec)):
            steps_per_sec[idx] = (steps_per_sec[idx])[:minlen]

        steps_per_sec = np.array(steps_per_sec)
        assert len(steps_per_sec.shape) == 2, steps_per_sec.shape
        print('steps_per_sec.shape: {}'.format(steps_per_sec.shape))
        game_info['steps_per_sec'] = steps_per_sec

    # Add last game.
    all_game_stats.append(game_info)
    print('\n\nDone printing, len all games: {}'.format(len(all_game_stats)))
    assert len(all_game_stats) == len(U.GAMES) == len(U.G_INDS_FAT)

    # --------------------------------------------------------------------------
    # Now go through this again, same logic, except plot.  Alphabetical order
    # from top row, w/one for legend to apply to subplots.  Remember, (average,
    # last_100), ideally report both.
    # --------------------------------------------------------------------------
    for game, (r,c) in zip(U.GAMES, U.G_INDS_FAT):
        idx = U.GAMES.index(game)
        s_stats = all_game_stats[idx]  # has current game statistics
        print(game, ': ', sorted(s_stats.keys()))

        nb_trials, _ = s_stats['steps_per_sec'].shape
        mean = np.mean(s_stats['steps_per_sec'], axis=0)
        std = np.std(s_stats['steps_per_sec'], axis=0)

        title_txt = '{} ({}x; {:.2f})'.format(game, nb_trials, mean[-1])
        ax[r,c].set_title(title_txt, fontsize=titlesize)
        ax[r,c].set_xlabel('Environment Steps (Millions)', fontsize=xsize)
        #ax[r,c].set_ylabel("Steps per Second".format(), fontsize=ysize)

        # Diving by 100 to make it go from 0 to 6
        xcoord = np.arange( len(mean) ) / 100
        ax[r,c].plot(xcoord, mean, lw=3, label='Env Steps per Sec\n(1 Step = 4 Frames)')
        ax[r,c].fill_between(xcoord, mean-std, mean+std,
                alpha=error_region_alpha)# facecolor=cc)

    # Put this on r=0, c=0, then hide it, just to get legend to appear.
    # Actually it hides the legend as well! But, fortunately it makes it
    # entirely white. That actually helps me!
    ax[0,0].set_visible(False)
    # Doesn't seem to work even if visibility above is set to True?
    #fig_info = 'From {}/{} to {}/{}'.format(MONTH_BEGIN, DAY_BEGIN, MONTH_END, DAY_END)
    #ax[0,0].text(x=0.5, y=0.5, s=fig_info, ha='center', va='center')

    # Bells and whistles
    for r in range(nrows):
        for c in range(ncols):
            #leg = ax[r,c].legend(loc="best", ncol=2, prop={'size':legendsize})
            #for legobj in leg.legendHandles:
            #    legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
            # I think it's better to share axes in the x direction to be
            # consistent with steps, but doing so removes the axis ticks. This
            # reverts it so we get the ticks on all the axis. Also for y.
            #ax[r,c].xaxis.set_tick_params(which='both', labelbottom=True)
            ax[r,c].yaxis.set_tick_params(which='both', labelleft=True)
            ax[r,c].set_ylim([0, 600])

    # Location (0,0) is bottom left. Doing (0,1) is upper left but the text
    # isn't visible (because `loc` is the lower left part of the legend).
    handles, labels = ax[r,c].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.030,0.65), prop={'size':legendsize})

    # Finally, save!! Can't do `.[...].png` since overleaf complains.
    plt.tight_layout()
    figname = 'fig_steps_per_sec_v01_case_{}.png'.format(CASE)
    plt.savefig(figname)
    print("Just saved:  {}".format(figname))


def _criteria(x):
    """Filter as usual.
    """
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


if __name__ == "__main__":
    # Iterate through all the *student* models and plot them.
    EXP_PATH = cfg.SNAPS_STUDENT
    dirs = sorted([join(EXP_PATH,x) for x in os.listdir(EXP_PATH) if _criteria(x)])
    print("Currently plotting with these models, one trained agent per file:")
    stats = defaultdict(list)

    for dd in dirs:
        last_part = os.path.basename(os.path.normpath(dd))
        if last_part in STUFF_TO_SKIP:
            print("  skipping {}".format(last_part))
            continue
        print("\nAnalyzing:   {}".format(dd))
        info = get_info(dd)
        key = '{}'.format(info['game_name'])

        ## # Do some extra corrections before adding to stats dict.
        ## # Because experiments 2 and 3 overlapped in terms of dates.
        ## if args.exp == 2:
        ##     if info['mb_start'] == 0.50:
        ##         print('  skipping {} due to mb_start {}'.format(key, info['mb_start']))
        ##         continue
        ## elif args.exp == 3:
        ##     if info['mb_start'] == 0.25:
        ##         print('  skipping {} due to mb_start {}'.format(key, info['mb_start']))
        ##         continue

        stats[key].append(info)

    report_combined_stats(stats)
