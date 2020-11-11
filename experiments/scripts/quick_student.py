"""Designed for fast inspection of standard trials involving STUDENTS.

Just run `python scripts/quick_student.py` to get figures saved in the student
directories (it doesn't aggregate them).

It's similar to `quick_dqn.py` except there is extra stuff we'd like to plot, in
particular to overlay the teacher vs student performance, and to show
statistics such as the data blend for each minibatch.

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

# -------------------------
# matplotlib stuff
titlesize = 33
xsize = 30
ysize = 30
ticksize = 25
legendsize = 23
# -------------------------

# Ignore these since I may have had outdated keys, etc. For example, in
# mid-March I got rid of 'num_episodes_around' and similar stuff. Also,
# sometimes training trials that are in progress don't have sufficient
# information for plotting.
STUFF_TO_SKIP = [
]


CONST = 1e6

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


def teacher_summary(ax,  exp_path, teacher_title, xvals, yvals, w, params,
                    i_steps, t_b):
    """For inspecting the matching process.

    As discussed in https://github.com/CannyLab/dqn/issues/9, I don't think we
    want the 'current' snapshots as that's confusing. Let's ignore it.

    June 2019: updated to handle the case of one_forward vs fixed_steps for the
    teacher overlap options, but maintains backwards compatibility. (So long as
    the teacher directory is unzipped and contains the v02 training summary
    file with the proper amount of lives recorded for each episode.)

    July 2019: fixing frames-per-ZPD issues.

    Parameters
    ----------
    xvals, yvals: the steps and (averaged) reward for the student.
    params: Dictionary of the information from the teacher.

    Returns
    -------
    x: the steps when a matching occurred, useful to plot as vertical lines.
    Edit: July 2019, now returning a dict to stuff as much stuff there as
    possible.
    """
    pth = join(exp_path,'teaching','teacher_0/summary.txt')
    with open(pth, 'r') as fh:
        match = json.load(fh)
        me = match['episodes']
        ms = match['matchings']
    num_steps  = params['teacher']['num_teacher_samples']
    num_ahead  = params['teacher']['num_snapshot_ahead']
    condense_f = params['teacher']['condense_freq']

    # File exists starting from June 12, 2019 and onwards. (Edit: actually maybe
    # not if we are just doing fixed steps ahead.)
    olap_pth = join(exp_path,'teaching','teacher_0/overlap.txt')
    if os.path.exists(olap_pth):
        with open(olap_pth, 'r') as fh:
            olap_stats_full = json.load(fh)
            olap_stats = olap_stats_full['overlap']
        # TODO we don't really use
        o_match_method   = params['teacher']['overlap']['match_method']
        o_prob_target    = params['teacher']['overlap']['prob_target']
        o_overlap_target = params['teacher']['overlap']['overlap_target']
        o_avg_class_prob = []
        o_avg_class_olap = []
        o_info_to_return = {
            'match_method': o_match_method,
            'prob_target': o_prob_target,
            'overlap_target': o_overlap_target,
        }
    else:
        olap_stats = None

    max_x = np.max(xvals) # for axis limitations
    idx = '0'
    x = []
    x2 = []  # ignore matches that don't give new zpd teacher
    current_num = []
    matched_num = []
    next_sn_num = []
    learner_rew = []
    current_rew = []
    matched_rew = []
    next_sn_rew = []

    # Now do episodes; for info about what we loaded from each snapshot.
    frames    = []
    ep_counts = []
    prog_mean = []
    prog_perp = []

    while True:
        if idx in ms:
            # 'matching' portion of the dictionary
            x.append( int(ms[idx]['learner_cum_steps']) )
            current_num.append( int(ms[idx]['previous_ss_num']) )
            matched_num.append( int(ms[idx]['matched_ss_num']) )
            next_sn_num.append( int(ms[idx]['zpd_ss_num']) )
            learner_rew.append( float(ms[idx]['learner_reward']) )
            current_rew.append( float(ms[idx]['previous_ss_reward']) )
            matched_rew.append( float(ms[idx]['matched_ss_reward']) )
            next_sn_rew.append( float(ms[idx]['zpd_ss_reward']) )

            # For handling last matched, we only want *new* ZPD snapshots.
            if len(x) == 1:
                assert len(next_sn_num) == 1
                x2.append( int(ms[idx]['learner_cum_steps']) )
            elif next_sn_num[-1] != next_sn_num[-2]:
                assert len(x) > 1
                x2.append( int(ms[idx]['learner_cum_steps']) )

            # 'episodes' portion of the dictionary
            total_s = int(me[idx]['episodes']['totals'])
            total_a = int(me[idx]['episodes']['total_actives'])
            totalf  = int(me[idx]['frames']['totals'])
            totala  = int(me[idx]['frames']['total_actives'])
            assert total_s == total_a, "{} vs {}".format(total_s, total_a)
            assert totalf == totala, "{} vs {}".format(totalf, totala)
            frames.append(totalf)
            ep_counts.append( int(me[idx]['episodes']['totals']) )
            if 'progress_scores' in me[idx]:
                prog_mean.append( float(me[idx]['progress_scores']['mean_progress_scores']) )
                prog_perp.append( float(me[idx]['progress_scores']['perplexity']) )

            # TODO Overlap stats -- makes a *second* figure.
            if olap_stats is not None:
                if 'avg_class_prob' in olap_stats[idx]:
                    assert o_match_method == 'one_forward', o_match_method
                    class_probs = olap_stats[idx]['avg_class_prob']
                    o_num_teachers = len(class_probs)
                if 'avg_olap_min_mean' in olap_stats[idx]:
                    assert o_match_method == 'train_net', o_match_method
                    olap_measures = olap_stats[idx]['avg_olap_min_mean']
                    o_num_teachers = len(olap_measures)

            # Proceed to next recorded data (i.e., the next *matching instance*).
            idx = str( int(idx)+1 )
        else:
            break
    #TODO
    #if olap_stats is not None:
    #    o_info_to_return['avg_class_prob'] = np.array(o_avg_class_prob)

    # Scale steps down to plot wrt millions of steps
    x = scale_steps(x)
    x2 = scale_steps(x2)
    ms = 10

    for step in x2:
        ax[1,0].axvline(x=step, color='orange', ls='--', lw=1.0)
    ax[1,0].set_title("Student Matching Instances ({} times)".format(len(learner_rew)),
        fontsize=titlesize)
    ax[1,0].set_xlabel("Student ENV Steps (M)", fontsize=xsize)
    ax[1,0].set_ylabel("Avg Past {} Rewards".format(w), fontsize=ysize)
    ax[1,0].plot(x, learner_rew, marker='x', ms=ms, color='orange',
        label='Learner Match Instances, last {:.1f}'.format(learner_rew[-1]))
    ax[1,0].plot(xvals, yvals, color='blue',
        label='Student Rew, last match @ {:.2f}M'.format(x[-1]))
    ax[1,0].set_xlim([0, max_x])

    ax[2,0].set_title("Teacher: {}".format(teacher_title), fontsize=titlesize)
    ax[2,0].set_xlabel("Student ENV Steps (M)", fontsize=xsize)
    ax[2,0].set_ylabel("Avg Past {} Rewards".format(w), fontsize=ysize)
    ax[2,0].plot(x, learner_rew, marker='x', ms=ms, color='black',
        label='Learner Rew, {:.1f} to {:.1f}'.format(learner_rew[0], learner_rew[-1]))
    ax[2,0].plot(x, matched_rew, marker='x', ms=ms, color='blue',
        label='Match Sshot, {:.1f} to {:.1f}'.format(matched_rew[0], matched_rew[-1]))
    ax[2,0].plot(x, next_sn_rew, marker='x', ms=ms, color='red',
        label='ZPD Sshot, {:.1f} to {:.1f}'.format(next_sn_rew[0], next_sn_rew[-1]))

    # Adjust based on the match method, if using code from June 12 2019 onwards.
    if olap_stats is None:
        match_title = "Condense {}, ahead {}, steps {}".format(
            condense_f, num_ahead, num_steps)
    else:
        if o_match_method == 'fixed_steps':
            match_title = "{} {}, cond {}, tstep {}".format(
                o_match_method, num_ahead, condense_f, num_steps)
        elif o_match_method == 'one_forward':
            match_title = "{} {}, nteach {}, tstep {}".format(
                o_match_method, o_prob_target, o_num_teachers, num_steps)
        elif o_match_method == 'train_net':
            match_title = "{} {}, nteach {}, tstep {}".format(
                o_match_method, o_overlap_target, o_num_teachers, num_steps)
        else:
            raise ValueError(o_match_method)
    ax[2,1].set_title(match_title, fontsize=titlesize)
    ax[2,1].set_xlabel("Student ENV Steps (M)", fontsize=xsize)
    ax[2,1].set_ylabel("RAW Teacher Sshot Index", fontsize=xsize) # NOT condensed!
    ax[2,1].plot(x, matched_num, marker='x', ms=ms, color='blue',
        label='Match Sshot Idx, {} to {}'.format(matched_num[0], matched_num[-1]))
    ax[2,1].plot(x, next_sn_num, marker='x', ms=ms, color='red',
        label='ZPD Sshot Idx, {} to {}'.format(next_sn_num[0], next_sn_num[-1]))

    # --------------------------------------------------------------------------
    # Plot number student steps while matched to a particular ZPD teacher.
    # Gives a rough sense of how long we're using a ZPD teacher, and we can
    # eyeball the teacher percent of the batch size to get the actual number of
    # teacher samples that were used in those steps. If we 'zig zag' by matching
    # back and forth, then we don't accumulate statistics from earlier matchings
    # (that'd make things a bit more complex, methinks). Also we're pretty much
    # guaranteed to get about the same amount of frames loaded (~50k), and that
    # info is in the plot axes. Huge note, does NOT include the last ZPD match
    # because that poses a number of administrative problems.
    # --------------------------------------------------------------------------
    assert len(frames) == len(x), "{} vs {}".format(len(frames), len(x))
    assert len(frames) >= len(x2), "{} vs {}".format(len(frames), len(x2))

    _cur_zpd = next_sn_num[0]
    _cur_stp = x[0]
    _steps = []
    _steps_raw = []
    for idx in range(1, len(next_sn_num)):
        _next_zpd = next_sn_num[idx]
        _stp = x[idx]
        if _next_zpd != _cur_zpd:
            data = (_cur_zpd, _stp - _cur_stp, _cur_stp)
            data_raw = (_cur_zpd, (_stp - _cur_stp) * CONST, _cur_stp*CONST)
            _steps.append( data )
            _steps_raw.append( data_raw )
            _cur_zpd = _next_zpd
            _cur_stp = _stp
    #for i1,i2 in zip(_steps,_steps_raw):
    #    print(i1,i2)

    # Back to plots!
    _xs = [z for (x,y,z) in _steps]
    _ys = [y for (x,y,z) in _steps]
    if len(_xs) == 0:
        _xs = [0]
        _ys = [0]
    ax[0,1].plot(_xs, _ys, marker='x', ms=ms, color='black',
        label = 'steps (millions) per zpd, ignore\n'
                'last zpd, so be careful w/results\n'
                'last step recorded @{:.2f}M\n'
                'avg / med / min / max / len'
                '\n{:.2f}, {:.2f}, {:.2f}, {:.2f}, {}'.format(
            _xs[-1], np.mean(_ys), np.median(_ys),
            np.min(_ys), np.max(_ys), len(_ys)))
    ax[0,1].set_title("Frm./Teach. {:.1f} ({}, {})".format(
        np.mean(frames), np.min(frames), np.max(frames)), fontsize=titlesize)
    ax[0,1].set_xlabel("Epis. per Teacher: {:.1f} +/- {:.1f}".format(
        np.mean(ep_counts), np.std(ep_counts)), fontsize=xsize)
    ax[0,1].set_ylabel("Stud. steps 'per' ZPD (omit last)".format(),
        fontsize=ysize)

    result = {
        'x': x,
        'x2': x2,
        'learner_rew': learner_rew,
        'matched_rew': matched_rew,
        'next_sn_rew': next_sn_rew,
        'next_sn_num': next_sn_num,
        'match_title': match_title,
    }
    if olap_stats is None:
        result['olap_stats'] = None
    else:
        result['olap_stats'] = o_info_to_return
    return result


def plot(exp_path, nrows=4, ncols=2, w=100):
    """Similar to DQN code, with extra stuff for teaching-related stats.

    The teacher's stuff is called later via `teacher_summary` and adds relevant
    information to subplots.

    Careful, this relies on assuming we have the teacher paths. I normally do
    `tar.gz` on the files so be sure to untar teachers as necessary. Also,
    `exp_path` is the full path to the learner agent.
    """
    figdir = join(exp_path,'figures')
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    title = U.get_title(exp_path)
    summary_train = U.load_summary_data(exp_path, train=True)
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
                           figsize=(11*ncols,8*nrows))

    # Plot rewards from training and testing runs. Use `raw_rew`, not `total_rew`.
    xvals, yvals = plot_one_history(ax[0,0], xkey='steps', ykey='raw_rew',
                data=summary_train, color='blue', label='Student', w=w)
    ax[0,0].set_xlabel("Student ENV Steps (M)", fontsize=xsize)
    ax[0,0].set_ylabel("Avg Past {} Rewards".format(w), fontsize=ysize)
    ax[0,0].set_title('{} (w={})'.format(title, w), fontsize=titlesize)

    # Also, overlay the expert! So, load teacher model, load path, then plot
    # data. Be careful we are allowed to do this 'substitution' to get the
    # expert data file ...
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
    plot_one_history(ax[0,0], xkey='steps', ykey='raw_rew', data=teacher_train,
            color='orange', label='Teacher', w=w)

    # Other interesting info that I saved from student's learning progress
    i_steps, info = U.load_info_data(exp_path)
    i_steps = scale_steps(i_steps)
    lr   = info['agent_lr'].values
    eps  = info['greedy_eps'].values
    mins = info['time_elapsed_mins'].values
    t_b  = info['teacher_blend'].values
    l_bm = info['bellman loss'].values
    if 'supervise loss' in info:
        l_sp = info['supervise loss'].values
        assert params['teacher']['supervise_loss']['enabled'], \
            'Did this student run terminate early?'
    else:
        l_sp = None
        assert not params['teacher']['supervise_loss']['enabled'], \
            'Did this student run terminate early?'
    if 'bellman from student' in info:
        bellman_student = info['bellman from student'].values
    else:
        bellman_student = None
    if 'bellman from teacher' in info:
        bellman_teacher = info['bellman from teacher'].values
    else:
        bellman_teacher = None

    # The teacher summary statistics -- I provide it the point when we last
    # match so we can easily tell how many steps correspond to a teacher.
    t_summary_res = teacher_summary(ax, exp_path, teacher_title, xvals=xvals,
                yvals=yvals, w=w, params=params, i_steps=i_steps, t_b=t_b)
    x_matches = t_summary_res['x']
    x_matches_2 = t_summary_res['x2']
    overlap_stats = t_summary_res['olap_stats']

    ax[1,1].set_xlabel("Student ENV Steps (M)", fontsize=xsize)
    ax[1,1].set_ylabel("Ratios/Fractions", fontsize=ysize)
    ax[1,1].set_title("Other Stats; Time {:.1f} m, {:.1f} h".format(
            mins[-1], mins[-1]/60), fontsize=titlesize)
    ax[1,1].plot(i_steps, eps, marker='x', color='blue',
            label="Student GreedyEps, last {:.2f}".format(eps[-1]))
    ax[1,1].plot(i_steps, t_b, marker='x', color='red',
            label="Teacher MB % Blend, last {:.2f}".format(t_b[-1]))

    # I added the weighted importance sampling on Feb 4.
    if 'beta_wis' in info:
        wis = info['beta_wis'].values
        ax[1,1].plot(i_steps, wis, marker='x', color='yellow',
                label="Beta Weight-IS, last {:.2f}".format(wis[-1]))

    # Extra losses for the student-teacher training. I think we can get away
    # with ignoring first element due to NaN in supervised loss. Also this is
    # only a subset of the minibatches, obviously.
    bs     = params['train']['batch_size']
    t_type = params['teacher']['supervise_loss']['type']
    t_lamb = params['teacher']['supervise_loss']['lambda']

    # --------------------------------------------------------------------------
    # Plot raw loss followed by log loss. The first item in steps has only
    # Bellman loss but it's 0 which skews the graph -- I think because the
    # network has only been initialized (not trained) so it predicts similar
    # stuff all around? For this and supervised loss, skip the first item.
    # Also, this has ALREADY been smoothed by 100 but I still find it's noisy
    # ... so let's smooth again by averaging over a small window?
    # --------------------------------------------------------------------------
    r = 3
    ww = 5

    # Raw Scale
    ax[r,0].set_title("Raw Loss, Every k MB (size {}), w={}".format(bs, ww),
                      fontsize=titlesize)
    ax[r,0].set_xlabel("Student ENV Steps (M)", fontsize=xsize)
    ax[r,0].set_ylabel("Losses", fontsize=ysize)
    label = 'Bell., avg {:.4f}, last {:.4f}'.format(np.mean(l_bm[1:]), l_bm[-1])
    ax[r,0].plot(i_steps[1:], U.smoothed(l_bm[1:], ww), color='red', label=label)
    if l_sp is not None:
        label = 'Supe., avg {:.4f}, last {:.4f}'.format(np.mean(l_sp[1:]), l_sp[-1])
        ax[r,0].plot(i_steps[1:], U.smoothed(l_sp[1:], ww), color='blue', label=label)
    if bellman_student is not None:
        label = 'Bellman S., avg {:.4f}, last {:.4f}'.format(
                np.mean(bellman_student[1:]), bellman_student[-1])
        ax[r,0].plot(i_steps[1:], U.smoothed(bellman_student[1:], ww),
                     color='darkblue', label=label)
    if bellman_teacher is not None:
        label = 'Bellman T., avg {:.4f}, last {:.4f}'.format(
                np.mean(bellman_teacher[1:]), bellman_teacher[-1])
        ax[r,0].plot(i_steps[1:], U.smoothed(bellman_teacher[1:], ww),
                     color='orange', label=label)
    # OK let's actually not do this, might be too many lines ...
    #for step in x_matches:
    #    ax[r,0].axvline(x=step, color='orange', ls='--', lw=1.0)

    # Log Scale
    imitation_title = "Log Scale w={}; ".format(ww)
    if l_sp is not None:
        imitation_title += "imit. {}, lambda {}".format(t_type, t_lamb)
    else:
        imitation_title += "imit. None"
    ax[r,1].set_title(imitation_title, fontsize=titlesize)
    _tf = params['train']['train_freq_per_step']  # vary up the titles
    ax[r,1].set_xlabel("train_freq_per_step: {}".format(_tf), fontsize=xsize)
    ax[r,1].set_ylabel("Losses (Log Scale)", fontsize=ysize)
    ax[r,1].plot(i_steps[1:], U.smoothed(l_bm[1:], ww), color='red',
                 label='Bellman (Log)'.format())
    if l_sp is not None:
        ax[r,1].plot(i_steps[1:], U.smoothed(l_sp[1:], ww), color='blue',
                     label='Supervised (Log)'.format())
    if bellman_student is not None:
        label = 'Bellman Student (Log)'.format()
        ax[r,1].plot(i_steps[1:], U.smoothed(bellman_student[1:], ww),
                     color='darkblue', label=label)
    if bellman_teacher is not None:
        label = 'Bellman Teacher (Log)'.format()
        ax[r,1].plot(i_steps[1:], U.smoothed(bellman_teacher[1:], ww),
                     color='orange', label=label)
    ax[r,1].set_yscale('log')
    for step in x_matches_2:
        ax[r,1].axvline(x=step, color='orange', ls='--', lw=1.0)

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

    # Finally, save!!
    plt.tight_layout()
    lastname = 'fig_train_results_{}.png'.format(title)
    figname = join(figdir,lastname)
    plt.savefig(figname)
    print("Just saved:  {}".format(figname))

    ## # Now do a second plot involving overlaps. Just in case. Not currently
    ## # using as of late July 2019, so here's just a code sketch.
    ## title = U.get_title(exp_path)
    ## nrows = 1
    ## ncols = 2
    ## fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
    ##                        figsize=(11*ncols,8*nrows))
    ## #class_probs shape (num_matching_times, num_teachers)
    ## #class_probs = overlap_stats['avg_class_prob']
    ## plt.tight_layout()
    ## lastname = 'fig_overlap_results_{}.png'.format(title)
    ## figname = join(figdir,lastname)
    ## plt.savefig(figname)
    ## print("Just saved:\n\t{}\n".format(figname))

    # ------------------------------------------------------------------------ #
    # Actually there is something I'd like to do: Make a smaller plot that
    # condenses some of the information into one row, for ease of reading
    # results. So keep this as one row, two columns.
    # ------------------------------------------------------------------------ #
    title = U.get_title(exp_path)
    summary_train = U.load_summary_data(exp_path, train=True)
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True,
                           figsize=(11*ncols,8*nrows))

    # Same as earlier.
    xvals, yvals = plot_one_history(ax[0,0], xkey='steps', ykey='raw_rew',
                data=summary_train, color='blue', label='Student', w=w)
    plot_one_history(ax[0,0], xkey='steps', ykey='raw_rew', data=teacher_train,
            color='orange', label='Teacher', w=w)
    if l_sp is not None:
        xtext = "ENV Steps (M), imit {}, lambda {}".format(t_type, t_lamb)
    else:
        xtext = "ENV Steps (M), imit None"
    ax[0,0].set_xlabel(xtext, fontsize=xsize-3)
    ax[0,0].set_ylabel("Avg Past {} Rewards".format(w), fontsize=ysize)
    ax[0,0].set_title('{} (w={})'.format(title, w), fontsize=titlesize)

    # Similar as earlier (in the teacher processing method, actually).
    x = t_summary_res['x']
    x2 = t_summary_res['x2']
    learner_rew = t_summary_res['learner_rew']
    matched_rew = t_summary_res['matched_rew']
    next_sn_rew = t_summary_res['next_sn_rew']
    next_sn_num = t_summary_res['next_sn_num']
    match_title = t_summary_res['match_title']
    ms = 10
    l1 = 'Learner Rew, {:.1f} to {:.1f}\n(last match @{:.2f}M)'.format(
            learner_rew[0], learner_rew[-1], x[-1])
    l2 = 'Match Sshot, {:.1f} to {:.1f}\n(matched {} times)'.format(
            matched_rew[0], matched_rew[-1], len(learner_rew))
    l3 = 'ZPD Sshot, {:.1f} to {:.1f}\nindices {} to {}'.format(next_sn_rew[0],
            next_sn_rew[-1], next_sn_num[0], next_sn_num[-1])
    ax[0,1].plot(x, learner_rew, marker='x', ms=ms, color='black', label=l1)
    ax[0,1].plot(x, matched_rew, marker='x', ms=ms, color='blue', label=l2)
    ax[0,1].plot(x, next_sn_rew, marker='x', ms=ms, color='red', label=l3)

    for step in x2:
        ax[0,1].axvline(x=step, color='orange', ls='--', lw=1.0)
    #second_title = "T {}; {:.1f}m, {:.1f}h".format(teacher_title, mins[-1], mins[-1]/60)
    second_title = "{}; {:.0f}m, {:.1f}h".format(teacher_title, mins[-1], mins[-1]/60)
    ax[0,1].set_title(second_title, fontsize=titlesize)
    ax[0,1].set_xlabel(match_title, fontsize=xsize-3)  # avoid duplicate text
    ax[0,1].set_ylabel("Avg Past {} Rewards".format(w), fontsize=ysize)

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
            ax[row,col].yaxis.set_tick_params(which='both', labelleft=True)

    plt.tight_layout()
    lastname = 'fig_train_results_{}_condensed.png'.format(title)
    figname = join(figdir,lastname)
    plt.savefig(figname)
    print("Just saved:  {}".format(figname))


if __name__ == "__main__":
    # Iterate through all the *student* models and plot them.
    EXP_PATH = cfg.SNAPS_STUDENT
    dirs = sorted([join(EXP_PATH,x) for x in os.listdir(EXP_PATH) \
            if U.criteria_for_quick_student(x)])
    print("Currently plotting with these models, one trained agent per file:")

    for dd in dirs:
        last_part = os.path.basename(os.path.normpath(dd))
        if last_part in STUFF_TO_SKIP:
            print("  skipping {}".format(last_part))
            continue
        print("\n\nAnalyzing:   {}".format(dd))
        plot(dd)
