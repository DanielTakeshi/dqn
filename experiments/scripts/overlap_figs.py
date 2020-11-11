"""Hopefully make it easier to plot overlap figures.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
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
np.set_printoptions(suppress=True, edgeitems=30, linewidth=190, precision=3)
CONST = 1e6

# ---------------------------------------------------------------------------- #
# matplotlib stuff
# ---------------------------------------------------------------------------- #
titlesize = 37
xsize = 33
ysize = 33
ticksize = 30
cbar_ticksize = 30
legendsize = 25
x_marker_size = 40
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# ADJUST !! Note: CASE will filter based on what experimental settings I used.
# BAR_CHARTS is because John wanted some bar charts instead of what I have here
# with a grid. I don't think it's going to be readable but I should try anyway.
# ---------------------------------------------------------------------------- #
CASE = 2
BAR_CHARTS = False
ADD_MARKERS = False
# ---------------------------------------------------------------------------- #

# For 4.0 experiments b/c I actually tried several.
OVERLAP_STUDENTS = {
    'alien':         '0.3',
    'beamrider':     '0.4',
    'boxing':        '0.1',
    'breakout':      '0.3',
    'pong':          '0.3',
    'qbert':         '0.1',
    'robotank':      '0.1',
    'seaquest':      '0.1',
    'spaceinvaders': '0.1',
}
GAME_DONE = {
    'alien':         False,
    'beamrider':     False,
    'boxing':        False,
    'breakout':      False,
    'pong':          False,
    'qbert':         False,
    'robotank':      False,
    'seaquest':      False,
    'spaceinvaders': False,
}

# ---------------------------------------------------------------------------- #
# Let's just specify the exact overlap agents to use in the _criteria method.
# Becuase we only need those, right? We can get info from the parameters file.
# ---------------------------------------------------------------------------- #

def _criteria(exp_path):
    """For the different stuff to plot:

    (See also: https://github.com/CannyLab/dqn/issues/66)

    - Experiment 4.0: use c2,c3 at the 0.25 blend, AND with dates before Aug 6.
      For these I actually did two random seeds each ... just pick the first
      one to plot? AH, we also have to filter by the overlap target type!

    - Ablation/Study 1: use c2,c3 at the 0.25 blend, AND with dates from Aug
      7-14 (I did overlaps after the Ablation 2 overlap studies, oops).

    - Ablation/Study 2: use c2,c3 at the 0.50 blend for the 250k/250k, 50/50 mb
      ablation study. These were all August 11-13.
    """
    def _date_ok(x, params, CASE):
        if CASE == 1:
            # For the original 4.0 experiments. Actually there's going to be two,
            # since I did 2x seeds, so pick one of them at random. Used 75/25 blend.
            MONTH_BEGIN, DAY_BEGIN = 7, 18
            MONTH_END, DAY_END = 8, 6
            name  = (params['env']['name']).replace('NoFrameskip-v4','').lower()
            otarget = str(params['teacher']['overlap']['overlap_target'])
            olap_filter = (otarget == OVERLAP_STUDENTS[name])
        elif CASE == 2:
            # Ablation 4.2 experiments, 250k/250k, 75/25 mb.
            MONTH_BEGIN, DAY_BEGIN = 8, 6
            MONTH_END, DAY_END = 8, 15
            olap_filter = True
        elif CASE == 3:
            # Ablation 4.2 experiments, 250k/250k, 50/50 mb (change below!).
            MONTH_BEGIN, DAY_BEGIN = 8, 11
            MONTH_END, DAY_END = 8, 15
            olap_filter = True
        else:
            raise ValueError(CASE)

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
        return begin and end and olap_filter

    # Filter the files. Don't forget to adjust `date_ok()` above as needed.
    last_part = os.path.basename(os.path.normpath(exp_path))
    if '_old' in last_part or 'setting' in last_part:
        return False
    with open(join(exp_path,'params.txt'), 'r') as f:
        params = json.load(f)
    c0 = _date_ok(last_part, params, CASE)
    c1 = params['teacher']['overlap']['match_method'] == 'train_net'
    if CASE == 1 or CASE == 2:
        c2 = params['teacher']['blend']['start'] == 0.25
        c3 = params['teacher']['blend']['end'] == 0.25
    elif CASE == 3:
        c2 = params['teacher']['blend']['start'] == 0.50
        c3 = params['teacher']['blend']['end'] == 0.50
    else:
        raise ValueError(CASE)
    c4 = (params['env']['name'])
    return (c0 and c1 and c2 and c3 and c4)


def scale_steps(x):
    x = np.array(x) / CONST
    return x


def get_overlap_matching_info(student_path):
    """Get the information about matching and overlap.

    The `student_path` contains the full path, not just the basename.  Returns
    a dict with as much info as possible.
    """
    with open(join(student_path, 'params.txt'), 'r') as f:
        params = json.load(f)
    num_steps        = params['teacher']['num_teacher_samples']
    num_ahead        = params['teacher']['num_snapshot_ahead']
    condense_f       = params['teacher']['condense_freq']
    o_match_method   = params['teacher']['overlap']['match_method']
    o_prob_target    = params['teacher']['overlap']['prob_target']
    o_overlap_target = params['teacher']['overlap']['overlap_target']

    # One of the two files in `teacher_0`, the other is overlap.
    pth = join(student_path, 'teaching', 'teacher_0/summary.txt')
    with open(pth, 'r') as fh:
        match = json.load(fh)
        me = match['episodes']
        ms = match['matchings']

    # --------------------------------------------------------------------------
    # The other file! Exists starting from June 12, 2019 and onwards, assuming
    # we do overlaps of course (which is by assumption here).  Relevant keys:
    # 'avg_olap_acc_mean' for debugging and 'avg_olap_min_mean' which is what
    # we really want. Collect results in numpy arrays.
    # --------------------------------------------------------------------------
    olap_pth = join(student_path, 'teaching', 'teacher_0/overlap.txt')
    assert os.path.exists(olap_pth), olap_pth
    with open(olap_pth, 'r') as fh:
        olap_stats_full = json.load(fh)
        olap_stats = olap_stats_full['overlap']  # only key
    o_accuracies = []
    o_measures = []

    # Now get the matching info, copied from prior code.
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

    while True:
        if idx in ms:
            # (1 of 2) 'matching' portion of the dictionary
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

            # (2 of 2) 'episodes' portion of the dictionary
            total_s = int(me[idx]['episodes']['totals'])
            total_a = int(me[idx]['episodes']['total_actives'])
            totalf  = int(me[idx]['frames']['totals'])
            totala  = int(me[idx]['frames']['total_actives'])
            assert total_s == total_a, "{} vs {}".format(total_s, total_a)
            assert totalf == totala, "{} vs {}".format(totalf, totala)
            frames.append(totalf)
            ep_counts.append( int(me[idx]['episodes']['totals']) )

            # Get corresponding overlap stats from the other file.
            assert 'avg_olap_min_mean' in olap_stats[idx], olap_stats[idx]
            assert o_match_method == 'train_net', o_match_method
            _o_measures = olap_stats[idx]['avg_olap_min_mean']
            _o_accuracies = olap_stats[idx]['avg_olap_acc_mean']
            o_measures.append(_o_measures)
            o_accuracies.append(_o_accuracies)
            assert len(_o_measures) == 23+1, len(_o_measures)

            # Proceed to next recorded data (i.e., the next *matching instance*).
            idx = str( int(idx)+1 )
        else:
            break

    # Scale steps down to plot wrt millions of steps
    x = scale_steps(x)
    x2 = scale_steps(x2)

    o_measures = np.array(o_measures)
    print('\n{}'.format(student_path))
    print('  len(x), all match cases: {}'.format(len(x)))
    print('  len(x2), only new ZPDs:  {}'.format(len(x2)))
    print('  len(matched_num): {}'.format(len(matched_num)))
    print('  o_measures: {}'.format(o_measures.shape))

    # Also, get ZPD teacher snapshot's rewards, straight from `quick_student.py`.
    teacher_models = params['teacher']['models']
    assert len(teacher_models) == 1, \
            "assume len(teacher_models) = 1, {}".format(len(teacher_models))
    s_last = os.path.basename(os.path.normpath(student_path))
    t_last = os.path.basename(os.path.normpath(teacher_models[0]))
    teacher_path = student_path.replace(s_last, t_last)
    teacher_path = teacher_path.replace('students/', 'teachers/')
    teacher_snap = U.load_snapshots_data(teacher_path)
    # Some of the older overlap trials I had didn't set condense_freq as 5.
    # But those should not affect correctness of the code logic.
    #assert params['teacher']['condense_freq'] == 5, params['teacher']['condense_freq']

    ret  = {
        'o_measures': o_measures,
        'matched_num': np.array(matched_num),
        'matched_rew': np.array(matched_rew),
        'learner_rew': np.array(learner_rew),
        'teacher_snap': teacher_snap,
        'next_sn_num': np.array(next_sn_num),
        'x': x,
        'x2': x2,
    }
    return ret, params


def plot_overlap(ax, g_idx, student_path, info, params):
    """Plot one student's overlap statistics.

    `info`: dict of information we want to plot, including overlap measures
    from every matching instance.

    Use `info['o_measures']` for data of shape (num_matching, num_snapshots
    +1), where we add one due to the student's dummy '-1' value at the end. The
    num_matching is how many times we had to perform the overlap test, which is
    triggered when the agent exceeds the reward of its current _matched_
    snapshot (not ZPD!).

    NOTE: I believe the learner will _match_ to any of the 117 (ignoring the
    last one, the 118th). For f_select corresponding to k-ahead, the ZPD
    teacher will only change every 5 (the condense_f) steps, but f_select for
    overlap means we may actually change the ZPD teacher more often. Or less
    often, if the same ZPD teacher is selected each time. I seriously doubt
    this is a big deal.

    See:
        https://matplotlib.org/gallery/  \
        images_contours_and_fields/image_annotated_heatmap.html  \
        #sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    for matplotlib examples.
    """
    r, c = U.G_INDS_SQUARE[g_idx]

    # Plot the overlap. Ignore last column, since that was the student index;
    # the goal is to compare student vs teachers, not the student vs itself!
    num_snapshots = 23
    num_matching, num_snapshots_p1 = info['o_measures'].shape
    assert num_snapshots+1 == num_snapshots_p1, num_snapshots_p1

    # --------------------------------------------------------------------------
    # We may want to subsample since some games have lots of matchings (60+).
    # As usual, np.linspace saves me some LeetCode-like coding. :-(
    # Ah, matched_{num,rew} include info from all 118 (or 117) snapshots, yet
    # we want the ones corresponding to the ZPD info.
    # --------------------------------------------------------------------------
    indices = [int(x) for x in np.linspace(start=0, stop=num_matching-1, num=30)]
    print('\nHere\'s the info to plot:')
    print(info['o_measures'][indices,:-1])
    print('matched_num: {}'.format(info['matched_num'][indices]))
    print('matched_rew: {}'.format(info['matched_rew'][indices]))
    print('learner_rew: {}'.format(info['learner_rew'][indices]))
    print('zpd_snp_num: {}'.format(info['next_sn_num'][indices]))
    info_to_use = info['o_measures'][indices,:-1]

    # Extract teacher snapshot rewards (that we actually used) in a list.
    teacher_snap = info['teacher_snap']
    idx_condensed = 1 + 5 * np.arange(1, num_snapshots+1)
    teacher_rew = teacher_snap['true_rew_epis'][idx_condensed].tolist()
    print('teacher_rew: {}'.format(teacher_rew))

    # --------------------------------------------------------------------------
    # NOTE: this part will add a bunch of 'x' markers. Make sure we want that.
    # --------------------------------------------------------------------------
    # Goal is to add an 'x' to any ZPD snapshot that would never get matched,
    # i.e., those with lower reward than the student, AND those with lower
    # index than the current matched snapshot. So all we see left in each row
    # is, at each matching query, the remaining, valid, ZPD candidates. See
    # `utils/summary.py` for how we do it in `dqn`.
    # --------------------------------------------------------------------------
    # To go from zpd snp idx to its index from 1 to 23, i.e.: 5 -> 1, 10 -> 2,
    # 15 -> 3, etc., just divide by five.
    # --------------------------------------------------------------------------
    for rr in range(len(indices)):
        if not ADD_MARKERS:
            continue
        l_rew = (info['learner_rew'][indices])[rr]
        m_rew = (info['matched_rew'][indices])[rr]
        m_num = (info['matched_num'][indices])[rr]
        zpd_n = (info['next_sn_num'][indices])[rr]
        for cc in range(num_snapshots):
            # First, remove from consideration indices <= matched num.
            # Second, remove any with less reward than learner.  Actually the
            # second condition only removes a handful of them.
            remove1 = (idx_condensed[cc] <= m_num)
            remove2 = (teacher_rew[cc] < l_rew)
            if remove1 and remove2:
                text = ax[r,c].text(cc, rr, 'x', color='red', ha='center',
                        va='center', fontsize=x_marker_size)
            if remove1 and not remove2:
                text = ax[r,c].text(cc, rr, 'x', color='blue', ha='center',
                        va='center', fontsize=x_marker_size)
            if not remove1 and remove2:
                text = ax[r,c].text(cc, rr, 'x', color='yellow', ha='center',
                        va='center', fontsize=x_marker_size)
            # cc is zero indexed.
            if cc+1 == int(zpd_n / 5.0):
                text = ax[r,c].text(cc, rr, 'o', color='orange', ha='center',
                        va='center', fontsize=x_marker_size)

    # Most values are less than 0.6, so let's set that as a uniform cutoff.
    im = ax[r,c].imshow(info_to_use, vmin=0, vmax=0.6, cmap='gray')

    # Sets color bars independently for each subplot.
    if c == 2:
        # https://stackoverflow.com/questions/18195758/
        # Actually this helps with making colorbars appear only per row.
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax[r,c])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        #cbar = ax[r,c].figure.colorbar(im, ax=ax[r,c])
        cbar = ax[r,c].figure.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('colorbar', rotation=-90, va='bottom')
        cbar.ax.tick_params(labelsize=cbar_ticksize)

    xlabels = [str(x+1) for x in np.arange(num_snapshots)]
    ax[r,c].set_xticks( np.arange(num_snapshots) )
    ax[r,c].set_xticklabels(xlabels)

    # Don't want all num_matches, just the `indices` which we're plotting.
    # And while we're at it, might plot max val as well.
    ylabels = indices
    for yy in range(len(ylabels)):
        ylabels[yy] = '{}, {:.2f}'.format(ylabels[yy], np.max(info_to_use[yy,:]))
    ax[r,c].set_yticks(np.arange(len(indices)))
    ax[r,c].set_yticklabels(ylabels)

    # Axes
    max_olap = np.max(info_to_use)
    last_part = os.path.basename(os.path.normpath(student_path))
    assert last_part[0] == '_', last_part
    agent = str(last_part[1:]).replace('snapshot_', '')
    overlap_target = params['teacher']['overlap']['overlap_target']
    subp_title = '{} OL {:.1f}'.format(agent, overlap_target)
    xtext = 'Teacher Snapshot Indices'.format()
    ytext = 'Calls to f_select ({} Times)'.format(num_matching)
    ax[r,c].set_xlabel(xtext, fontsize=xsize)
    ax[r,c].set_ylabel(ytext, fontsize=ysize)
    ax[r,c].set_title(subp_title, fontsize=titlesize)


def plot(dirs, nrows=3, ncols=3, w=100):
    """Plot a single figure (and may a few others) related to overlap.

    A tall plot:  nrows=5, ncols=2. Figsize: 20*ncols, 10*nrows?
    A square one: nrows=3, ncols=3. Figsize: xx*ncols, xx*nrows?

    The `dirs` contains a list of all the full path to the students that were
    trained via a ZPD matching process that utilizes overlap.
    """
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=False,
            sharey=False, figsize=(15*ncols,19*nrows),
            gridspec_kw={'width_ratios': [1, 1, 1.08]})

    for g_idx,student_path in enumerate(dirs):
        game = U.GAMES[g_idx].lower()
        assert game in student_path, '{} {}'.format(game, student_path)
        # Get information about the overlaps and matching process.
        info, params = get_overlap_matching_info(student_path)
        plot_overlap(ax, g_idx, student_path, info, params)

    for r in range(nrows):
        for c in range(ncols):
            # No legends here.
            #leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            #for legobj in leg.legendHandles:
            #    legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
            # I think it's better to share axes in the x direction to be
            # consistent with steps, but doing so removes the axis ticks. This
            # reverts it so we get the ticks on all the axis.
            ax[r,c].xaxis.set_tick_params(which='both', labelbottom=True)
            ax[r,c].yaxis.set_tick_params(which='both', labelleft=True)

    plt.tight_layout()
    if CASE == 1:
        figname = 'fig_overlap_all_games_75-25-1M-50k.png'
    elif CASE == 2:
        figname = 'fig_overlap_all_games_75-25-250k-250k.png'
    elif CASE == 3:
        figname = 'fig_overlap_all_games_50-50-250k-250k.png'
    else:
        raise ValueError(CASE)
    plt.savefig(figname)
    print("Just saved:  {}".format(figname))


if __name__ == "__main__":
    # Iterate through all the *student* models and plot them.
    EXP_PATH = cfg.SNAPS_STUDENT
    dirs = sorted(
        [join(EXP_PATH,x) for x in os.listdir(EXP_PATH) if _criteria(join(EXP_PATH,x))]
    )

    # Remove duplicates.
    actual_dirs = []
    for dd in dirs:
        last_part = os.path.basename(os.path.normpath(dd))
        assert last_part[0] == '_', last_part
        last_part = last_part[1:]
        game = (last_part.split('_'))[0].lower()
        if not GAME_DONE[game]:
            actual_dirs.append(dd)
            GAME_DONE[game] = True

    print("Will plot with these student models:")
    for dd in actual_dirs:
        print("  {}".format(dd))
    print('')
    plot(actual_dirs)
