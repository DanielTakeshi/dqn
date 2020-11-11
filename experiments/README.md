# Configurations and Scripts for Experiments

This contains:

- A `settings` folder with a bunch of files for running experiments. **See
  `settings/README.md` for a detailed overview of the parameters.**

- A `scripts` folder for plotting and analyzing results.

- A `notebooks` folder for any jupyter notebooks we might want to use.,

- A bunch of scripts (e.g., `atari.py`) that can do teacher-only or
  student-teacher training, depending on the configuration file we pass as
  input.

Here, we go over how to use the code, logging, plotting, and networks/models.

- [Quick Usage](#quick-usage)
- [Logging](#logging)
- [Plotting](#plotting)
- [Networks and Models](#networks-and-models)

**Note**: if you are getting an error related to a missing file in the
`episodes/` directory, you likely need to run this:

```
python scripts/make_summary_v02_file.py  --agent _pong_standard_2019-03-20-10-29_s7827
```

to get the correct summary file. I am going to run this on all the agents I
have, then re-compress into tar.gz format, and send them to other machines.
That way, we don't run into this issue again.


## Quick Usage


For a teacher-only run:

```
python atari.py pong_standard
```

For a student-teacher run:

```
python atari.py pong_snapshot
```

Where the `pong_standard` and `pong_snapshot` may vary across games.

Output is currently saved in the directory specified by
[common_config.py](https://github.com/CannyLab/dqn/blob/master/dqn/common_config.py).

For now, we don't use `atari-multi-agents.py` or `atari-multi-teachers.py`, but
I think we may start using the latter two if we train a single student with
multiple teachers.



## Zero vs One Indexing

The indexing convention (0 vs 1) used to be a huge problem. For now, we are
standardizing things so that inside the code, we do *0 indexing only for all
our indices*.

However, when we save or load *episodes* (really, lifespans) or *snapshots*, it
is necessary to use ONE-indexing for those. Thus, keep all values 0-indexed
until just before we have to load. Then we add +1. Since we saved all these
pickle files, it would be a hassle to write a script that tries to re-name
them. Note that the snapshot and lifespan pickle files numbers are padded with
zeros, to get 4 digits for snapshots and 7 digits for lifespans.

Thus, these debug messages are all 1-indexed for lives and episodes:

```
05-06 13:04:33 train_env   : DEBUG    Episode 1 done, 733 steps, 165.00 raw reward
05-06 13:04:33 train_replay: DEBUG    Life 4 of length 0 added into replay, with 4/4 active lifespans, 725 total frames and 725 registered frames.
05-06 13:04:34 train_env   : DEBUG    Life 4: 246/971 steps, 5.00 rewards, 4.25 mean rewards, speed 597.25 frames/second, epsilon 1.00
05-06 13:04:34 train_replay: DEBUG    Life 4 finished w/246 transition frames. RBuffer contains 4/4 active episodes, 971 total frames and 971 registered frames.
05-06 13:04:34 train_replay: DEBUG    Life 5 of length 0 added into replay, with 5/5 active lifespans, 971 total frames and 971 registered frames.
05-06 13:04:34 train_env   : DEBUG    Life 5: 109/1080 steps, 0.00 rewards, 3.40 mean rewards, speed 604.26 frames/second, epsilon 1.00
05-06 13:04:34 train_replay: DEBUG    Life 5 finished w/109 transition frames. RBuffer contains 5/5 active episodes, 1080 total frames and 1080 registered frames.
05-06 13:04:34 train_replay: DEBUG    Life 6 of length 0 added into replay, with 6/6 active lifespans, 1080 total frames and 1080 registered frames.
05-06 13:04:34 train_env   : DEBUG    Life 6: 71/1151 steps, 1.00 rewards, 3.00 mean rewards, speed 613.93 frames/second, epsilon 1.00
05-06 13:04:34 train_replay: DEBUG    Life 6 finished w/71 transition frames. RBuffer contains 6/6 active episodes, 1151 total frames and 1151 registered frames.
05-06 13:04:34 train_env   : DEBUG    Episode 2 done, 434 steps, 80.00 raw reward
```

The only other case where I think we have one-indexing is when we load the
"train v02" episode summary file, which has life begin and life end keys:

```
      life_idx_begin  life_idx_end  life_num  raw_rew  steps
0                  1             1         1      -21    847
1                  2             2         1      -21    882
2                  3             3         1      -21    987
3                  4             4         1      -20    924
```

Thus, index 0 is when we actually use for episode 1, but if we load
`life_idx_begin` or `life_idx_end` we have to be careful since those are
1-indexed.




## Logging

Each trained agent (either a teacher or a student) should have its own directory
where stuff is saved. For teacher-only training, it looks like this, saved in
a `teacher/` directory:

```
(py3.5-pytorch) seita@triton2:/nfs/diskstation/seita/mt-dqn/teachers$ ls _pong_standard_2019-03-20-10-29_s7827/
total 3.7M
-rw-rw-r-- 1 nobody nogroup   13K Mar 20 14:32 download_summary.txt
drwxrwxr-x 1 nobody nogroup  154K Mar 20 14:32 episodes
-rw-rw-r-- 1 nobody nogroup  164K Mar 20 14:32 events.out.tfevents.1553102960.triton2
drwxrwxr-x 1 nobody nogroup    82 Mar 20 14:34 figures
-rw-rw-r-- 1 nobody nogroup  131K Mar 20 14:32 info_summary.txt
drwxrwxr-x 1 nobody nogroup    22 Mar 20 10:31 learner_replay
drwxrwxr-x 1 nobody nogroup     0 Mar 20 10:28 learner_snapshots
-rw-rw-r-- 1 nobody nogroup  211K Mar 20 14:32 main.prof
-rw-rw-r-- 1 nobody nogroup   72K Mar 20 14:32 monitor.csv
-rw-rw-r-- 1 nobody nogroup  2.1K Mar 20 10:28 params.txt
-rw-rw-r-- 1 nobody nogroup  157K Mar 20 14:32 play.prof
-rw-rw-r-- 1 nobody nogroup  2.0M Mar 20 14:32 root.log
drwxrwxr-x 1 nobody nogroup  4.9K Mar 20 14:32 snapshots
drwxrwxr-x 1 nobody nogroup     0 Mar 20 10:29 teaching
-rw-rw-r-- 1 nobody nogroup 1006K Mar 20 14:32 train.log
(py3.5-pytorch) seita@triton2:/nfs/diskstation/seita/mt-dqn/teachers/settings$
```

For student-teacher training, saved in a `student/` directory:

```
(py3.5-pytorch) seita@BID-Biye-v2:~/dqn/experiments$ ls /data/mt-dqn/students/_breakout_snapshot_2019-03-20-10-22_s94500/
total 117M
-rw-rw-r-- 1 seita seita  13K Mar 20 15:56 download_summary.txt
drwxrwxr-x 2 seita seita 4.0K Mar 20 10:22 episodes
-rw-rw-r-- 1 seita seita  94M Mar 20 15:56 events.out.tfevents.1553102573.BID-Biye-v2
drwxrwxr-x 2 seita seita 4.0K Mar 20 16:54 figures
-rw-rw-r-- 1 seita seita 192K Mar 20 15:56 info_summary.txt
drwxrwxr-x 2 seita seita 4.0K Mar 20 10:24 learner_replay
drwxrwxr-x 2 seita seita 4.0K Mar 20 15:24 learner_snapshots
-rw-rw-r-- 1 seita seita 313K Mar 20 15:56 main.prof
-rw-rw-r-- 1 seita seita 132K Mar 20 15:56 monitor.csv
-rw-rw-r-- 1 seita seita 2.8K Mar 20 10:22 params.txt
-rw-rw-r-- 1 seita seita 162K Mar 20 15:56 play.prof
-rw-rw-r-- 1 seita seita  18M Mar 20 15:56 root.log
drwxrwxr-x 2 seita seita 4.0K Mar 20 15:56 snapshots
drwxrwxr-x 3 seita seita 4.0K Mar 20 10:22 teaching
-rw-rw-r-- 1 seita seita 4.5M Mar 20 15:56 train.log
(py3.5-pytorch) seita@BID-Biye-v2:~/dqn/experiments$
```

It's mostly similar. Highlights:

- `info_summary.txt` records statistics for plotting later.

- `episodes` saves all the lifespans from teacher-only training, so we can load
  them into student-teacher later. This is the most memory intensive part of
  training. **This is one of the more confusing parts, admittedly -- each pickle
  file saved is a LIFESPAN, which could mean one episode for one-life games, but
  e.g., for Breakout with five lives, one episode covers five consecutive
  lifespans**. There are `*.txt` files to help understand the results.

- `*.log` files are files for the logger, `*.prof` are for profiling,
  `params.txt` saves exact parameters so we can tell what we trained with. We
  actually have two of these logs because (for some reason) we call
  multiprocessing code.

- `monitor.csv` is from OpenAI-based monitoring and helps us identify true
  rewards and true episode steps, rather than clipped rewards, etc.

- `snapshots/` saves snapshots from the teacher, so we can match for
  student-teacher training later.

- `teaching/teacher_k` for teacher index `k` is for statistics on
  student-teacher training wrt teacher `k`. Not used for teacher-only training.

- Use `figures/` after the training, for inserting plots.

The other stuff, I don't use.

*Note on profiling*: we have code for profiling, and it has been very helpful.
See https://github.com/CannyLab/dqn/issues/28 for details. To enable or disable
it, look at two spots in the code: `atari.py` where we call the main method, and
in `dqn/processes.py` at the play and play profiler methods. In both files, I
have one of the two cases (use or don't use profiling) comment out. Feel free to
switch them as needed.


## Plotting

Run:

```
python scripts/quick_dqn.py
```

and

```
python scripts/quick_student.py
```

These cycle through the `teacher` and `student` directories, respectively, and
plot automatically. For each agent, it saves in a `figures/` directory. There
are also scripts that combine results, which we will need for a final paper to
concisely express results.



## Networks and Models

The `atari.py` main method basically has the full DQN pipeline there, except it
calls a whole bunch of stuff deeper into the code. It's useful that things are
there, e.g., it's easy to print the model:

```
AtariNet(
  (conv1): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
  (conv2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
  (conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
  (fc4): Linear(in_features=3136, out_features=256, bias=True)
  (fc5): Linear(in_features=256, out_features=6, bias=True)
)
```

which is similar as those used in prior works, though with perhaps a different
number of fully connected features. (Edit: actually, in general we use 512 for
the fully connected layer, which is consistent with DeepMind's earlier work). We
do not support dueling architectures or prioritized experience replay.

Allen defined another model architecture, for progress nets but it's similar:

```
AtariProgressNet(
  (conv1): ModuleList(
    (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
    (1): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
  )
  (conv2): ModuleList(
    (0): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    (1): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
  )
  (conv3): ModuleList(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
  )
  (fc4): ModuleList(
    (0): Linear(in_features=3136, out_features=256, bias=True)
    (1): Linear(in_features=3136, out_features=256, bias=True)
  )
  (fc5): ModuleList(
    (0): Linear(in_features=256, out_features=6, bias=True)
    (1): Linear(in_features=256, out_features=6, bias=True)
  )
)
```

But for now, if we use uniform sampling from the teacher, just ignore this. The
computation of progress scores can take a while and it's a confounding effect;
let's not test it just yet.
