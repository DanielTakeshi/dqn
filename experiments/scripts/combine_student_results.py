"""
Run `python scripts/combine_student_results.py` to coalesce student results
together.  This contrasts with `scripts/quick_student.py`, which was designed
to quickly cycle through students and save *one figure per student*.

This will also generate a figure to coalesce these results together. See:
https://github.com/DanielTakeshi/IL_ROS_HSR/blob/master/scripts/bedmake_box_plots.py
for potentially similar code.

This uses BAR CHARTS.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import offsetbox
from matplotlib.ticker import FuncFormatter
import argparse, csv, math, os, pickle, sys, inspect, json
from os.path import join
import numpy as np
import pandas as pd
from dqn import common_config as cfg
from collections import defaultdict
import utils as U
plt.style.use('seaborn-darkgrid')
sns.set_style("darkgrid")
np.set_printoptions(linewidth=180, edgeitems=10)

# ------------------------------------------------------------------------------
# matplotlib stuff
# ------------------------------------------------------------------------------
titlesize = 55
xsize = 46
ysize = 46
ticksize = 46
legendsize = 48
student_colors = ['black', 'red', 'blue', 'purple', 'brown']
teacher_color = 'orange'
error_region_alpha = 0.25
bwidth = 0.3
# ------------------------------------------------------------------------------
CONST = 1e6


def scale_steps(x):
    x = np.array(x) / CONST
    return x


def get_reward_info(xkey, ykey, data, w):
    """Get information about rewards during training.

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
    #_label = 'len/#eps: {},\n(sm) avg/max/last: {:.1f}, {:.1f}, {:.1f}'.format(
    #        len(xdata), np.mean(xdata), np.max(xdata), xdata[-1])

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
    #_label += "\n(raw) avg1,2/last100: {:.1f}, {:.1f}, {:.1f}".format(
    #        np.mean(_y), np.mean(yy), np.mean(_y[-100:]))
    info = {
        'avg_all_episodes':  np.mean(_y),
        'avg_after_explore': np.mean(yy),
        'avg_last_100':      np.mean(_y[-100:]),
    }
    return info


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
    s_info = get_reward_info(xkey='steps', ykey='raw_rew', data=summary_train, w=w)

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
    t_info = get_reward_info(xkey='steps', ykey='raw_rew', data=teacher_train, w=w)

    # Other interesting info that I saved from student's learning progress.
    # If an error results, check that the run didn't terminate early, etc.
    i_steps, info = U.load_info_data(exp_path)
    i_steps = scale_steps(i_steps)

    # Ah this is going to be a bit annoying but w/e, b/c one of Pong failed.
    if '_pong_snapshot_2019-08-22-21-57_s64329' in exp_path:
        print('  At the problematic: _pong_snapshot_2019-08-22-21-57_s64329')
        print('  That one exited early due to Pong-specific stuff.')
    else:
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

    result = {
        'game_name':        U.get_game_name(s_last),
        's_avg_all_epis':   s_info['avg_all_episodes'],
        's_avg_after_expl': s_info['avg_after_explore'],
        's_avg_last_100':   s_info['avg_last_100'],
        't_avg_all_epis':   t_info['avg_all_episodes'],
        't_avg_after_expl': t_info['avg_after_explore'],
        't_avg_last_100':   t_info['avg_last_100'],
        'overlap_param':    t_overlap_p,
        'match_method':     t_overlap_m,
        'supervise_lambda': t_lambda,
        'mb_start':         params['teacher']['blend']['start'],
        'mb_end':           params['teacher']['blend']['end'],
        'train_freq':       params['train']['train_freq_per_step'],
    }
    return result


def report_combined_stats(stats, args):
    """Report combined stats, ideally for a plot.

    :param stats: dict, with key --> list, where the list has one item per
        random seed. This helps us combine results more easily.
    """
    if args.exp == 1:
        U.G_OLAP = U.G_OLAP_01
    elif args.exp == 2:
        U.G_OLAP = U.G_OLAP_02
    elif args.exp == 3:
        U.G_OLAP = U.G_OLAP_03
    elif args.exp == 4:
        U.G_OLAP = U.G_OLAP_04
    elif args.exp == 5:
        U.G_OLAP = U.G_OLAP_05
    else:
        raise ValueError(args.exp)

    nrows = 2
    ncols = 5
    w = 100
    # Increase factor to `nrows` to make plot 'taller'.
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=False,
                           figsize=(11*ncols,9*nrows))

    # Data for plots later
    all_game_stats = []  # list of dicts, one per game
    t_stats = defaultdict(list)

    # Go through and print text I can copy/paste, while adding to the plot.
    game_idx = 0
    print('\n\n\t\tNEW GAME: {}'.format(U.GAMES[game_idx]))
    game_info = {}

    avg_best = -np.inf
    avg_key = None
    last_best = -np.inf
    last_key = None

    for key in sorted(stats.keys()):
        game = U.GAMES[game_idx]
        if game.lower() not in key:
            game_idx += 1
            game = U.GAMES[game_idx]

            # Before we do that, report which methods are best:
            print('\nDone with a game!')
            print('  avg: {:.1f}  {}'.format(avg_best, avg_key))
            print('  last: {:.1f} {}'.format(last_best, last_key))
            avg_best = -np.inf
            avg_key = None
            last_best = -np.inf
            last_key = None

            print('\n\n\t\tNEW GAME: {}'.format(game))
            # Add the previously accumulated states to the game_stats.
            all_game_stats.append(game_info)
            game_info = {}

        num_trials = len(stats[key])
        print('\n{}   len(stats[key]): {}'.format(key, num_trials))
        s_all_epis   = [x['s_avg_all_epis']   for x in stats[key]]
        s_after_expl = [x['s_avg_after_expl'] for x in stats[key]]
        s_last_100   = [x['s_avg_last_100']   for x in stats[key]]
        t_all_epis   = [x['t_avg_all_epis']   for x in stats[key]]
        t_after_expl = [x['t_avg_after_expl'] for x in stats[key]]
        t_last_100   = [x['t_avg_last_100']   for x in stats[key]]
        assert len(set(t_all_epis)) == 1,   t_all_epis
        assert len(set(t_after_expl)) == 1, t_after_expl
        assert len(set(t_last_100)) == 1,   t_last_100
        print('                Student --- Teacher')
        s_after_expl = np.mean(s_after_expl)
        t_after_expl = np.mean(t_after_expl)
        s_last_100   = np.mean(s_last_100)
        t_last_100   = np.mean(t_last_100)
        print('avg after expl: {:.1f}   {:.1f}'.format(s_after_expl, t_after_expl))
        print('last 100 avg:   {:.1f}   {:.1f}'.format(s_last_100,   t_last_100))

        if s_after_expl > avg_best:
            avg_best = s_after_expl
            avg_key = key
        if s_last_100 > last_best:
            last_best = s_last_100
            last_key = key

        # Add teacher stats, should match for all in this loop so we just do once.
        if len(t_stats[game]) == 0:
            t_stats[game].append( (t_after_expl, t_last_100) )

        # Only want student samples for statistics that we will actually be using.
        info = key.split('__')
        if info[1] == 'fixed_steps':
            #assert num_trials == args.num_trials, num_trials
            if num_trials != args.num_trials:
                print('WARNING! we have {} trials, but should have {}'.format(
                        num_trials, args.num_trials))
            num_ahead = info[2]
            game_info[num_ahead] = (s_after_expl, s_last_100)
        elif info[1] == 'train_net':
            # Each game has a certain value we might want
            target = info[2]
            if target != U.G_OLAP[game_idx]:
                continue
            assert num_trials == args.num_trials, num_trials
            # Could use `target` as key if we want multiple overlaps.
            game_info['ol'] = (s_after_expl, s_last_100)
        else:
            raise ValueError(info)

    # Add last game.
    print('\nDone with a game!')
    print('  avg: {:.1f}  {}'.format(avg_best, avg_key))
    print('  last: {:.1f} {}'.format(last_best, last_key))

    all_game_stats.append(game_info)
    print('\n\nDone printing, len all games: {}'.format(len(all_game_stats)))
    assert len(all_game_stats) == len(U.GAMES) == len(U.G_INDS_FAT) == len(U.G_OLAP)

    # --------------------------------------------------------------------------
    # Now go through this again, same logic, except plot.  Alphabetical order
    # from top row, w/one for legend to apply to subplots.  Remember, (average,
    # last_100), ideally report both.
    # --------------------------------------------------------------------------
    x = np.arange(6)
    for game, (r,c), gol in zip(U.GAMES, U.G_INDS_FAT, U.G_OLAP):
        ax[r,c].set_title('{}'.format(game), fontsize=titlesize)
        # This text can take up some real estate!
        #ax[r,c].set_ylabel("Reward".format(), fontsize=ysize)
        idx = U.GAMES.index(game)
        s_stats = all_game_stats[idx]  # has current game statistics
        print(game, ': ', sorted(s_stats.keys()))

        # TEMPORARY HACKS
        if 'ol' not in s_stats:
            s_stats['ol'] = [0, 0]
        if '00' not in s_stats:
            s_stats['00'] = [0, 0]
        if '02' not in s_stats:
            s_stats['02'] = [0, 0]
        if '05' not in s_stats:
            s_stats['05'] = [0, 0]
        if '10' not in s_stats:
            s_stats['10'] = [0, 0]
        if '-1' not in s_stats:
            s_stats['-1'] = [0, 0]
        if '-2' not in s_stats:
            s_stats['-2'] = [0, 0]

        ydata_avg = [
            s_stats['-1'][0],
            s_stats['-2'][0],
            s_stats['00'][0],
            s_stats['02'][0],
            s_stats['05'][0],
            s_stats['10'][0],
        ]
        ydata_last = [
            s_stats['-1'][1],
            s_stats['-2'][1],
            s_stats['00'][1],
            s_stats['02'][1],
            s_stats['05'][1],
            s_stats['10'][1],
        ]
        slabel0 = '(4:1) Student Average'
        slabel1 = '(4:1) Student Last 100'
        if args.exp == 4 or args.exp == 5:
            slabel0 = '(2:1) Student Average'
            slabel1 = '(2:1) Student Last 100'
        rects0 = ax[r,c].bar(x,
                height=ydata_avg,
                width=bwidth,
                alpha=0.9,
                color='red',
                label=slabel0)
        rects1 = ax[r,c].bar(x+bwidth,
                height=ydata_last,
                width=bwidth,
                alpha=0.9,
                color='blue',
                label=slabel1)
        #ax[r,c].set_xticklabels( ('0A', '2A', '5A', '10A', 'BA', 'L{}'.format(gol)) )
        ax[r,c].set_xticklabels( ('BA', 'RA', '0A', '2A', '5A', '10A'.format()) )
        ax[r,c].set_xticks( x+0.1 ) # tune offset to get it centered

        # Horizontal line for teacher performance? Not sure ...
        assert len(t_stats[game]) == 1, len(t_stats[game])
        t_avg, t_last = t_stats[game][0]
        ax[r,c].axhline(y=t_avg, linestyle='--', lw=6, color='red',
                        label='Teacher Average')
        ax[r,c].axhline(y=t_last, linestyle='--', lw=6, color='blue',
                        label='Teacher Last 100')

    # Put this on r=0, c=0, then hide it, just to get legend to appear.
    # Actually it hides the legend as well! But, fortunately it makes it
    # entirely white. That actually helps me!
    ax[0,0].set_visible(False)

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
            # reverts it so we get the ticks on all the axis.
            #ax[r,c].xaxis.set_tick_params(which='both', labelbottom=True)

    handles, labels = ax[r,c].get_legend_handles_labels()
    #for legobj in handles:
    #    legobj.set_linewidth(5.0)
    # Location (0,0) is bottom left. Doing (0,1) is upper left but the text
    # isn't visible (because `loc` is the lower left part of the legend).
    fig.legend(handles, labels, loc=(0.005,0.65), prop={'size':legendsize})

    # Finally, save!! Can't do `.[...].png` since overleaf complains.
    plt.tight_layout()
    if args.exp == 1:
        figname = 'fig_combo_results_exp01_1M_50k.png'.format()
    elif args.exp == 2:
        figname = 'fig_combo_results_exp02_mb_75-25_updates_4-1.png'.format()
    elif args.exp == 3:
        figname = 'fig_combo_results_exp03_mb_50-50_updates_4-1.png'.format()
    elif args.exp == 4:
        figname = 'fig_combo_results_exp04_mb_50-50_updates_2-1.png'.format()
    elif args.exp == 5:
        figname = 'fig_combo_results_exp05_mb_25-75_updates_2-1.png'.format()
    else:
        raise ValueError(args.exp)
    plt.savefig(figname)
    print("Just saved:  {}".format(figname))


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # ADJUST THE SETTINGS WHICH WE FILTER. It's similar to the plotting script,
    # except here we have to iterate through all paths first to combine them
    # into groups. Then we report the average statistics.
    # --------------------------------------------------------------------------
    # Note: for exp's 3 to 5, I did random ahead, and those we should use
    # instead of the overlaps -- that way we can *justify* results with overlap.
    # --------------------------------------------------------------------------
    EXP_PATH = cfg.SNAPS_STUDENT
    pp = argparse.ArgumentParser()
    pp.add_argument('--exp', type=int)
    args = pp.parse_args()
    if args.exp is None:
        raise ValueError('Pass in the --exp type.')

    # --------------------------------------------------------------------------
    # The number of trials that I (should) have per experiment. I'll throw some
    # debug messages in case we're missing some.
    # --------------------------------------------------------------------------
    if args.exp == 1:
        # 1M / 50k split, 75-25 mb, 4:1 step:update
        args.num_trials = 2
    elif args.exp == 2:
        # 250k / 250k, 75-25 mb, 4:1 step:update
        args.num_trials = 1
    elif args.exp == 3:
        # 250k / 250k, 50-50 mb, 4:1 step:update (report this!)
        args.num_trials = 2
    elif args.exp == 4:
        # 250k / 250k, 50-50 mb, 2:1 step:update (report this!)
        args.num_trials = 2
    elif args.exp == 5:
        # 250k / 250k, 25-75 mb, 2:1 step:update (oops)
        args.num_trials = 2
    else:
        raise ValueError(args.exp)

    # Iterate through all the *student* models.
    dirs = sorted( [join(EXP_PATH,x) for x in os.listdir(EXP_PATH) \
            if U._criteria_for_experiments(x,args)] )
    print("Currently plotting with these models, one trained agent per file:")
    stats = defaultdict(list)

    for dd in dirs:
        last_part = os.path.basename(os.path.normpath(dd))
        if last_part in U.STUFF_TO_SKIP:
            print("  skipping {} due to STUFF_TO_SKIP".format(last_part))
            continue
        print("\nAnalyzing:   {}".format(dd))
        info = get_info(dd)
        key = '{}__{}__{}'.format(info['game_name'], info['match_method'],
                info['overlap_param'])

        # Do some extra corrections before adding to stats dict.
        # Because experiments 2 and 3 overlapped in terms of dates.
        # Also, `info['mb_start']` refers to percentage of TEACHER samples.
        tf = info['train_freq']

        if args.exp == 1:
            if info['mb_start'] != 0.25:
                print('  skipping {} due to mb_start {}'.format(key, info['mb_start']))
                continue
            if tf != 4:
                print('  skipping {} due to train_freq {}'.format(key, tf))
                continue

        elif args.exp == 2:
            if info['mb_start'] != 0.25:
                print('  skipping {} due to mb_start {}'.format(key, info['mb_start']))
                continue
            if tf != 4:
                print('  skipping {} due to train_freq {}'.format(key, tf))
                continue

        elif args.exp == 3:
            if info['mb_start'] != 0.50:
                print('  skipping {} due to mb_start {}'.format(key, info['mb_start']))
                continue
            if tf != 4:
                print('  skipping {} due to train_freq {}'.format(key, tf))
                continue
            if info['match_method'] == 'train_net':
                print('  skipping {} due to train_net (no overlap for now)'.format(key))
                continue

        elif args.exp == 4:
            if info['mb_start'] != 0.50:
                print('  skipping {} due to mb_start {}'.format(key, info['mb_start']))
                continue
            if tf != 2:
                print('  skipping {} due to train_freq {}'.format(key, tf))
                continue
            if info['match_method'] == 'train_net':
                print('  skipping {} due to train_net (no overlap for now)'.format(key))
                continue

        elif args.exp == 5:
            if info['mb_start'] != 0.75:
                print('  skipping {} due to mb_start {}'.format(key, info['mb_start']))
                continue
            if tf != 2:
                print('  skipping {} due to train_freq {}'.format(key, tf))
                continue
            if info['match_method'] == 'train_net':
                print('  skipping {} due to train_net (no overlap for now)'.format(key))
                continue

        stats[key].append(info)

    report_combined_stats(stats, args)
