"""This combines a bunch of learning curves for all the games.

For bar charts, see `combine_student_results.py`.
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
titlesize = 53
xsize = 42
ysize = 42
ticksize = 42
legendsize = 48
scolors = ['gold', 'red', 'blue', 'purple', 'silver', 'orange']
tcolor = 'black'
error_region_alpha = 0.25
bwidth = 0.3
slw = 7
# ------------------------------------------------------------------------------
CONST = 1e6
LEN_REWARDS = 596 # this should be the length ...


def scale_steps(x):
    x = np.array(x) / CONST
    return x


def get_info(exp_path, w=100):
    """Gather information, in a similar manner as scripts/quick_student.py.
    """
    title = U.get_title(exp_path)
    summary_train = U.load_summary_data(exp_path, train=True)
    s_steps, s_info = U.load_info_data(exp_path)
    s_reward = s_info['true_avg_rew'].values

    # Ah this is going to be a bit annoying but w/e, b/c one of Pong failed.
    if '_pong_snapshot_2019-08-22-21-57_s64329' in exp_path:
        print('  At the problematic: _pong_snapshot_2019-08-22-21-57_s64329')
        print('  That one exited early due to Pong-specific stuff.')
        s_steps = np.array([(x*10000+50000) for x in range(LEN_REWARDS)])
        tmp = np.ones((LEN_REWARDS,)) * s_reward[-1]
        for i in range(len(s_reward)):
            tmp[i] = s_reward[i]
        s_reward = tmp

    s_steps = scale_steps(s_steps)
    assert len(s_steps) == len(s_reward)

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

    # CANNOT DO THIS FOR EARLIER RUNS, dating back to before the summer, I think.
    #t_steps, t_info = U.load_info_data(teacher_path)

    # AH, we did not record 'true_avg_rew' in the teacher ... ugh.  So for this
    # just read the root file and parse like I do here. That gives us the same
    # values that I use for the 'true_avg_rew' key.
    t_steps = []
    t_reward = []
    teacher_root_file = join(teacher_path, 'root.log')
    with open(teacher_root_file, 'r') as f:
        for line in f:
            if 'completed' in line and '**********' in line and 'steps' in line:
                linesp = line.split()
                assert linesp[0] == '**********', linesp
                assert linesp[2] == 'steps', linesp
                steps = int(linesp[1])
                t_steps.append(steps)
            if 'Last 100 results: avg' in line:
                linesp = line.split()
                assert linesp[0] == 'Last', linesp
                assert linesp[1] == '100', linesp
                assert linesp[2] == 'results:', linesp
                assert linesp[3] == 'avg', linesp
                assert ',' in linesp[4], linesp
                rew = float(linesp[4].strip(','))
                t_reward.append(rew)
    t_steps = scale_steps(t_steps)
    assert len(t_steps) == len(t_reward)

    # More annoying stuff ...
    if len(s_steps) > LEN_REWARDS:
        print('for {}, len(s_steps) = {} so chopping to {}'.format(
                    exp_path, len(s_steps), LEN_REWARDS))
        s_steps = s_steps[:LEN_REWARDS]
        s_reward = s_reward[:LEN_REWARDS]
    if len(t_steps) > LEN_REWARDS:
        print('for {}, len(t_steps) = {} so chopping to {}'.format(
                    exp_path, len(t_steps), LEN_REWARDS))
        t_steps = t_steps[:LEN_REWARDS]
        t_reward = t_reward[:LEN_REWARDS]
    assert len(s_steps) == LEN_REWARDS, len(s_steps)
    assert len(s_reward) == LEN_REWARDS, len(s_reward)
    assert len(t_steps) == LEN_REWARDS, len(t_steps)
    assert len(t_reward) == LEN_REWARDS, len(t_reward)

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
        'overlap_param':    t_overlap_p,
        'match_method':     t_overlap_m,
        'supervise_lambda': t_lambda,
        'student_rew':      s_reward,   # student reward every 10k steps (starts @ 50k)
        'teacher_rew':      t_reward,   # teacher reward every 10k steps (starts @ 50k)
        'student_steps':    s_steps,    # should be same among all trials but save anyway
        'teacher_steps':    t_steps,    # should be same among all trials but save anyway
        'mb_start':         params['teacher']['blend']['start'],
        'mb_end':           params['teacher']['blend']['end'],
        'train_freq':       params['train']['train_freq_per_step'],
    }
    return result


def _get_array(list_of_items):
    nb = len(list_of_items)
    lengths = [len(x) for x in list_of_items]
    if len(lengths) > 1 and np.std(lengths) > 0:
        print('Error with lengths: {}'.format(lengths))
        sys.exit()
    return np.array(list_of_items)


def report_combined_stats(stats, args, square=True):
    """Report combined stats, ideally for a plot.

    :param stats: dict, with key --> list, where the list has one item per
        random seed. This helps us combine results more easily.

    Also `square` to make it a 3x3 grid which is more readable.
    """
    if args.exp == 1:
        G_OLAP = U.G_OLAP_01
    elif args.exp == 2:
        G_OLAP = U.G_OLAP_02
    elif args.exp == 3:
        G_OLAP = U.G_OLAP_03
    elif args.exp == 4:
        G_OLAP = U.G_OLAP_04
    else:
        raise ValueError(args.exp)

    # Increase factor to `nrows` to make plot 'taller'.
    if not square:
        # Old way:
        nrows = 2
        ncols = 5
        fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=False,
                               figsize=(11*ncols,9*nrows))
        INDICES = U.G_INDS_FAT
    else:
        # New way:
        nrows = 4  # blank row, make invisible, move legend here!
        ncols = 3
        fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=False,
                               figsize=(12*ncols,9*nrows),
                               gridspec_kw={'height_ratios': [5,5,5,1]})
        INDICES = U.G_INDS_SQUARE
    w = 100

    # Data for plots later
    all_game_stats = []  # list of dicts, ONE PER GAME
    t_stats = defaultdict(list)

    # Go through and print text I can copy/paste, while adding to the plot.
    # Unlike earlier, game_info (and t_stats) needs to have the x coordinates,
    # since we're doing full learning curves.
    game_idx = 0
    print('\n\n\t\tNEW GAME: {}'.format(U.GAMES[game_idx]))
    game_info = {}  # For each game, collect stats, put in `all_game_stats`.

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
        s_rews = _get_array([x['student_rew'] for x in stats[key]])
        t_rews = _get_array([x['teacher_rew'] for x in stats[key]])
        print('student/teacher rewards: {} {}'.format(s_rews.shape, t_rews.shape))
        #print('std(student): {}'.format(np.std(s_rews, axis=0)))
        #print('std(teacher): {}'.format(np.std(t_rews, axis=0)))
        assert np.max( np.abs(np.std(t_rews,axis=0)) ) < 0.001, \
                'We are using the same teacher, right? The StDev should be zero.'
        assert num_trials == s_rews.shape[0] == t_rews.shape[0], num_trials

        # Let's not do this in case we want to plot standard deviation
        #s_rews = np.mean(s_rews, axis=0)

        # Eh this could easily be a global list since all the games use the
        # same number of steps (thus far) but this may give us flexibility later.
        s_steps = np.mean(_get_array([x['student_steps'] for x in stats[key]]), axis=0)
        t_steps = np.mean(_get_array([x['teacher_steps'] for x in stats[key]]), axis=0)

        # Add teacher stats, should match for all in this loop so we just do once.
        t_rews = np.mean(t_rews, axis=0)

        if len(t_stats[game]) == 0:
            t_stats[game].append( (t_steps,t_rews) )

        # Only want student samples for statistics that we will actually be using.
        info = key.split('__')
        if info[1] == 'fixed_steps':
            #assert num_trials == args.num_trials, num_trials
            if num_trials != args.num_trials:
                print('WARNING! we have {} trials, but should have {}'.format(
                        num_trials, args.num_trials))
            num_ahead = info[2]
            game_info[num_ahead] = (s_steps,s_rews)
        elif info[1] == 'train_net':
            # Each game has a certain value we might want
            target = info[2]
            if target != U.G_OLAP[game_idx]:
                continue
            assert num_trials == args.num_trials, num_trials
            # Could use `target` as key if we want multiple overlaps.
            game_info['ol'] = (s_steps,s_rews)
        else:
            raise ValueError(info)

    # Add last game.
    all_game_stats.append(game_info)
    print('\n\nDone printing, len all games: {}'.format(len(all_game_stats)))
    assert len(all_game_stats) == len(U.GAMES) == len(U.G_INDS_FAT) == len(G_OLAP)

    # --------------------------------------------------------------------------
    # Now go through this again, same logic, except plot.  Alphabetical order
    # from top row, w/one for legend to apply to subplots.
    # --------------------------------------------------------------------------
    for game, (r,c), gol in zip(U.GAMES, INDICES, G_OLAP):
        ax[r,c].set_title('{}'.format(game), fontsize=titlesize)
        idx = U.GAMES.index(game)
        s_stats = all_game_stats[idx]  # has current game statistics
        print(game, ': ', sorted(s_stats.keys()))

        # Just take first one b/c they are all the same.
        t_x, t_y = t_stats[game][0]
        ax[r,c].plot(t_x, t_y, lw=10, ls='--', color=tcolor, label='DDQN Teacher')

        # --------------------------------------------------------------------------
        # NOTE: adjust this based on how many of the student 'keys' I want to
        # post.  Then this will require an additional adjustment over the
        # legends. Note that we remove the first item due to game name.
        # --------------------------------------------------------------------------
        key = '-1'
        if key in s_stats:
            s_x, s_y = s_stats[key]
            s_y = np.mean(s_y, axis=0)
            ax[r,c].plot(s_x, s_y, lw=slw, color=scolors[0], label='S, Best Ahead')

        key = '-2'
        if key in s_stats:
            s_x, s_y = s_stats[key]
            s_y = np.mean(s_y, axis=0)
            ax[r,c].plot(s_x, s_y, lw=slw, color=scolors[1], label='S, Random Ahead')

        key = '00'
        if key in s_stats:
            s_x, s_y = s_stats[key]
            s_y = np.mean(s_y, axis=0)
            ax[r,c].plot(s_x, s_y, lw=slw, color=scolors[2], label='S, 0 Ahead')

        key = '02'
        if key in s_stats:
            s_x, s_y = s_stats[key]
            s_y = np.mean(s_y, axis=0)
            ax[r,c].plot(s_x, s_y, lw=slw, color=scolors[3], label='S, 2 Ahead')

        key = '05'
        if key in s_stats:
            s_x, s_y = s_stats[key]
            s_y = np.mean(s_y, axis=0)
            ax[r,c].plot(s_x, s_y, lw=slw, color=scolors[4], label='S, 5 Ahead')

        key = '10'
        if key in s_stats:
            s_x, s_y = s_stats[key]
            s_y = np.mean(s_y, axis=0)
            ax[r,c].plot(s_x, s_y, lw=slw, color=scolors[5], label='S, 10 Ahead')
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------

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

    if not square:
        # Put this on r=0, c=0, then hide it, just to get legend to appear.
        # Actually it hides the legend as well! But, fortunately it makes it
        # entirely white. That actually helps me!
        ax[0,0].set_visible(False)

        handles, labels = ax[1,1].get_legend_handles_labels()
        #for legobj in handles:
        #    legobj.set_linewidth(5.0)
        # Location (0,0) is bottom left. Doing (0,1) is upper left but the text
        # isn't visible (because `loc` is the lower left part of the legend).
        fig.legend(handles, labels, loc=(0.005,0.550), prop={'size':legendsize})
    else:
        # Global legend.
        #fig.subplots_adjust(bottom=0.3)
        #leg = ax.flatten()[-2].legend(loc='lower center', ncol=2,
        #            prop={'size':legendsize})
        #for legobj in leg.legendHandles:
        #    legobj.set_linewidth(5.0)

        # Eh not working let's just do what I did earlier.
        ax[3,0].set_visible(False)
        ax[3,1].set_visible(False)
        ax[3,2].set_visible(False)
        handles, labels = ax[0,0].get_legend_handles_labels()
        # If doing '2A, 5A' etc labels.
        #leg = fig.legend(handles, labels, loc=(0.12,0.00), ncol=4,
        # prop={'size':legendsize})
        # If doing 2 Ahead, etc.
        leg = fig.legend(handles, labels, loc=(0.08,0.00), ncol=4,
                        prop={'size':legendsize})
        for legobj in leg.legendHandles:
            legobj.set_linewidth(13)

    # Finally, save!! Can't do `.[...].png` since overleaf complains.
    plt.tight_layout()
    if args.exp == 1:
        figname = 'fig_combo_lcurves_exp01_1M_50k.png'.format()
    elif args.exp == 2:
        figname = 'fig_combo_lcurves_exp02_mb_75-25_updates_4-1.png'.format()
    elif args.exp == 3:
        figname = 'fig_combo_lcurves_exp03_mb_50-50_updates_4-1.png'.format()
    elif args.exp == 4:
        figname = 'fig_combo_lcurves_exp04_mb_50-50_updates_2-1.png'.format()
    else:
        raise ValueError(args.exp)
    plt.savefig(figname)
    print("Just saved:  {}".format(figname))

    # NEW Sept 8, now figure out the spread of values between games.
    print('\nSpread of values:')
    last = {}
    last_ratio = {}
    keys = ['-2', '00', '02', '05', '10']
    idx_max = -1
    for game, (r,c), gol in zip(U.GAMES, INDICES, G_OLAP):
        idx = U.GAMES.index(game)
        s_stats = all_game_stats[idx]  # has current game statistics
        values_l = []
        for k in keys:
            s_x, s_y = s_stats[k]
            s_y = np.mean(s_y, axis=0) # last value for the past 100 eps.
            values_l.append( s_y[-1] )
        last[game] = np.array(values_l)

        # Compare with the -1 ahead.
        s_x, s_y = s_stats['-1']
        s_y = np.mean(s_y, axis=0)
        best_ahead_val = s_y[-1]

        imax = np.argmax(last[game])
        best_teach_val = values_l[imax]
        ratio = best_teach_val / best_ahead_val
        last_ratio[game] = ratio

        print('  On game {}'.format(game))
        print('{}    w/spread {:.2f}'.format(values_l, np.std(values_l)))
        print('best teaching value: {:.1f} at key {}'.format(best_teach_val,
            keys[imax]))
        print('the best ahead got: {:.1f}'.format(best_ahead_val))
        print('ratio: {:.2f}'.format(ratio))
    
    print('\nFor all the games:')
    for key in sorted(last_ratio.keys()):
        print('{}:  {:2f}'.format(key, last_ratio[key]))


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # ADJUST THE SETTINGS WHICH WE FILTER. It's similar to the plotting script,
    # except here we have to iterate through all paths first to combine them
    # into groups. Then we report the average statistics.
    # --------------------------------------------------------------------------
    EXP_PATH = cfg.SNAPS_STUDENT
    pp = argparse.ArgumentParser()
    pp.add_argument('--exp', type=int)
    args = pp.parse_args()
    if args.exp is None:
        raise ValueError('Pass in the --exp type.')

    # The number of trials that I have per experiment.
    # See `combine_student_results.py` for what these mean.
    if args.exp == 1:
        args.num_trials = 2
    elif args.exp == 2:
        args.num_trials = 1
    elif args.exp == 3:
        args.num_trials = 2
    elif args.exp == 4:
        args.num_trials = 2
    else:
        raise ValueError(args.exp)

    # Iterate through all the *student* models and plot them.
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
        tf = info['train_freq']

        if args.exp == 2:
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

        stats[key].append(info)

    report_combined_stats(stats, args)
