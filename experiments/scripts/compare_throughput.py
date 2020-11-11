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


def _info_for_plots(stats, t_stats, target_num_trials=2):
    """Go through and collect data for one experimental condition.

    Calling this method several times means we should be able to compare many
    different settings. Unlike earlier, game_info (and t_stats) needs to have
    the x coordinates, since we're doing full learning curves.

    Returns a list that has all the game stats we want. It should be a list
    with ONE ITEM PER GAME, so a length 9 list here!
    """
    all_game_stats = []
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
            if num_trials != target_num_trials:
                print('WARNING! we have {} trials, but should have {}'.format(
                        num_trials, target_num_trials))
            num_ahead = info[2]
            game_info[num_ahead] = (s_steps,s_rews)
        elif info[1] == 'train_net':
            continue
        else:
            raise ValueError(info)

    # Add last game.
    all_game_stats.append(game_info)
    print('\n\nDone printing, len all games: {}'.format(len(all_game_stats)))
    assert len(all_game_stats) == len(U.GAMES) == len(U.G_INDS_FAT)
    return all_game_stats


def report_combined_stats(stats_3, stats_4, args, w=100):
    """Report combined stats, ideally for a plot.

    :param stats: dict, with key --> list, where the list has one item per
        random seed. This helps us combine results more easily.
    """
    # Increase factor to `nrows` to make plot 'taller'.
    nrows = 2
    ncols = 5
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=False,
                           figsize=(11*ncols,8*nrows))
                           #gridspec_kw={'height_ratios': [5,5,5,1]})
    INDICES = U.G_INDS_FAT

    # Teacher data for plots later.
    t_stats_3 = defaultdict(list)
    t_stats_4 = defaultdict(list)

    # Do what I did earlier, except for BOTH of the stats here. Yeah !!
    print('\n*************************************************')
    print('COLLECTING DATA FROM FIRST EXPERIMENTAL CONDITION')
    print('*************************************************\n')
    all_game_stats_3 = _info_for_plots(stats=stats_3, t_stats=t_stats_3)
    print('\n*************************************************')
    print('COLLECTING DATA FROM FIRST EXPERIMENTAL CONDITION')
    print('*************************************************\n')
    all_game_stats_4 = _info_for_plots(stats=stats_4, t_stats=t_stats_4)

    # --------------------------------------------------------------------------
    # Plot experiment condition 3 and 4 on the same plot. The shape of `s_y`
    # here, i.e., the reward, is (num_trials, num_recorded) so we could do that
    # as standard deviation, but might be noisy ... also these ALREADY include
    # an implicit smoothing over the past 100 episodes.
    # --------------------------------------------------------------------------
    def _plot(r, c, key, s_stats_3, s_stats_4, color, label, force_color=False,
            std_curves=False):
        # Case 1, try to plot everything together w/same color codes:
        if False:
            s_x, s_y = s_stats_3[key]
            s_y = np.mean(s_y, axis=0)
            ax[r,c].plot(s_x, s_y, ls='--', lw=slw, color=color, label=label+', 4:1')
            s_x, s_y = s_stats_4[key]
            s_y = np.mean(s_y, axis=0)
            ax[r,c].plot(s_x, s_y, lw=slw, color=color, label=label+', 2:1')
        # Case 2, try to use standard deviations?
        if True:
            if force_color:
                cc = 'gold'
            else:
                cc = 'blue'
            s_x, s_y = s_stats_3[key]
            s_y = np.mean(s_y, axis=0)
            ax[r,c].plot(s_x, s_y, lw=slw, color=cc, label=label+', 4:1')
            if std_curves:
                ax[r,c].fill_between(s_x,
                                     s_y+np.std(s_y, axis=0),
                                     s_y-np.std(s_y, axis=0),
                                     color=cc,
                                     alpha=error_region_alpha)
            if force_color:
                cc = 'orange'
            else:
                cc = 'red'
            s_x, s_y = s_stats_4[key]
            s_y = np.mean(s_y, axis=0)
            ax[r,c].plot(s_x, s_y, lw=slw, color=cc, label=label+', 2:1')
            if std_curves:
                ax[r,c].fill_between(s_x,
                                     s_y+np.std(s_y, axis=0),
                                     s_y-np.std(s_y, axis=0),
                                     color=cc,
                                     alpha=error_region_alpha)

    # --------------------------------------------------------------------------
    # Now go through this again, same logic, except plot.  Alphabetical order
    # from top row, w/one for legend to apply to subplots.
    # --------------------------------------------------------------------------
    for game, (r,c) in zip(U.GAMES, INDICES):
        ax[r,c].set_title('{}'.format(game), fontsize=titlesize)
        idx = U.GAMES.index(game)

        # Keys: ['-1', '-2', '00', '02', '05', '10'] where -1 and -2 are BA and RA.
        print('\nKeys for s_stats_3, and then s_stats_4:')
        s_stats_3 = all_game_stats_3[idx]
        print(game, ': ', sorted(s_stats_3.keys()))
        s_stats_4 = all_game_stats_4[idx]
        print(game, ': ', sorted(s_stats_4.keys()))

        # Just take first one b/c teacher stats should be the same. Actually
        # wait maybe we don't need the teacher here? Think about it ...
        t_x, t_y = t_stats_3[game][0]
        if True:
            ax[r,c].plot(t_x, t_y, lw=10, ls='--', color=tcolor, label='DDQN Teacher')
        _t_x, _t_y = t_stats_4[game][0]
        assert np.allclose(t_x, _t_x), '{} {}'.format(t_x, _t_x)
        assert np.allclose(t_y, _t_y), '{} {}'.format(t_y, _t_y)

        # --------------------------------------------------------------------------
        # NOTE: adjust based on how many of the student 'keys' I want to post.
        # Toggle which ones we want on/off. SAME COLOR CODE AS PRIOR FIGURE, if
        # we are using all select functions.  But we prob. don't need best
        # ahead. Honestly it seems best just to let ONE be used at a time.
        # --------------------------------------------------------------------------
        if True:
            key = '-1'
            _plot(r, c, key, s_stats_3, s_stats_4, scolors[0], label='S, Best Ahead',
                    force_color=True)
        if False:
            key = '-2'
            _plot(r, c, key, s_stats_3, s_stats_4, scolors[1], label='S, Rand Ahead')
        if False:
            key = '00'
            _plot(r, c, key, s_stats_3, s_stats_4, scolors[2], label='S, 0 Ahead')
        if False:
            key = '02'
            _plot(r, c, key, s_stats_3, s_stats_4, scolors[3], label='S, 2 Ahead')
        if False:
            key = '05'
            _plot(r, c, key, s_stats_3, s_stats_4, scolors[4], label='S, 5 Ahead')
        if True:
            key = '10'
            _plot(r, c, key, s_stats_3, s_stats_4, scolors[5], label='S, 10 Ahead')
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

    # Put this on r=0, c=0, then hide it, just to get legend to appear.
    ax[0,0].set_visible(False)
    handles, labels = ax[1,1].get_legend_handles_labels()
    # Location (0,0) is bottom left. Doing (0,1) is upper left but the text
    # isn't visible (because `loc` is the lower left part of the legend).
    fig.legend(handles, labels, loc=(0.005,0.500), prop={'size':legendsize})

    # Finally, save!! Can't do `.[...].png` since overleaf complains.
    plt.tight_layout()
    figname = 'fig_throughput_student.png'.format()
    plt.savefig(figname)
    print("Just saved:  {}".format(figname))


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # NOW WE ASSUME WE'RE COMPARING EXP's 3 AND 4.
    # --------------------------------------------------------------------------
    EXP_PATH = cfg.SNAPS_STUDENT
    pp = argparse.ArgumentParser()
    args = pp.parse_args()
    args.num_trials_exp_3 = 2
    args.num_trials_exp_4 = 2

    # Iterate through all the *student* models.
    dirs = sorted( [join(EXP_PATH,x) for x in os.listdir(EXP_PATH) \
            if U._criteria_for_experiments_throughput(x,args)] )
    print("Currently plotting with these models, one trained agent per file:")

    stats_3 = defaultdict(list)
    stats_4 = defaultdict(list)

    for dd in dirs:
        last_part = os.path.basename(os.path.normpath(dd))
        if last_part in U.STUFF_TO_SKIP:
            print("  skipping {} due to STUFF_TO_SKIP".format(last_part))
            continue
        print("\nAnalyzing:   {}".format(dd))
        info = get_info(dd)
        key = '{}__{}__{}'.format(info['game_name'], info['match_method'],
                info['overlap_param'])
        mb = info['mb_start']
        tf = info['train_freq']
        mm = info['match_method']

        # We only want experiments 3 and 4.
        if mb == 0.50 and tf == 4 and mm != 'train_net':
            stats_3[key].append(info)
        elif mb == 0.50 and tf == 2 and mm != 'train_net':
            stats_4[key].append(info)
        else:
            print('  skipping {},  mm,tf,mm: {}, {}, {}'.format(key, mb,tf,mm))
            continue

    print('\nNow going to report on all these stats.')
    print('  len stats 3, 4 dicts: {} and {}'.format(len(stats_3), len(stats_4)))
    print('')
    report_combined_stats(stats_3, stats_4, args)
