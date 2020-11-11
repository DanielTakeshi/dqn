# Configuration Files Documentation

Anything of the form `<game>_standard.json` is standard DDQN (teacher-only
training in my terminology). For Pong, also have an extra `fast` since we tuned
the hyperparameters, but we also keep a non-fast version for consistency --- and
we probably should stop using the fast version anyway. For teachers, use
`<game>_snapshot.json`.

All scripts for the 6 games (and each for Pong) should be standardized with
similar settings. The `global_settings.json` is loaded by default, and settings
within each `.json` file will override global settings if they are specified.



## Ordinary Training Parameters

For agent training: `num_steps` indicates the `n` we're using for `n`-step
returns. We don't support combining both for now. The others are
straightforward, with the exception that `target_sync_per_step` in practice is
really 4x what we put in the parameter. So, if we use 2500, then with respect to
actual steps taken (not frames but steps) we do a sync every 10000 steps.

```
"train": {
    "num_steps": 1,
    "hidden_size": 512,
    "double": true,
    "target_sync_per_step": 2500,
    "train_freq_per_step": 4,
    "gamma": 0.99,
    "batch_size": 32
},
```

## Teacher Parameters

For anything related to student-teacher training, we have "teacher parameters"
which are here:

```
"teacher": {
    "type": "snapshots",
    "progress_measure": "projection",
    "teacher_samples_uniform": true,
    "overlap": {
        "match_method": "fixed_steps",
        "prob_target": 0.3,
        "overlap_target": 0.3,
        "pretrained_model": null
    },
    "models": ["_pong_standard_fast_2019-02-28-19-27_s16434"],
    "batch_size": 128,
    "num_snapshot_ahead": 10,
    "condense_freq": 5,
    "negative_correction": false,
    "progress_epsilon": 0.000001,
    "temperature": 1,
    "dedup_iterations": 0,
    "init_wis_beta": 0.0,
    "replay_size": 1e6,
    "num_teacher_samples": 50000,
    "blend": {
      "frames": 100,
      "start": 0.5,
      "end": 0.5
    },
    "supervise_loss": {
      "type": "margin",
      "enabled": true,
      "lambda": 0.100,
      "margin": 0.8
    }
}
```

In order (roughly) here's what the (hyper)parameters mean, with **TODO**s marked
if I'm not entirely sure or want to investigate further:

- For `type` and `progress_measure`, use "snapshots" and "projection"
  respectively. There are others but I'm not sure what those mean. **TODO**

- The `match_method` was recently added (June 2019) to handle different ways
  for determining matching. Look at `SnapshotsTeacher` for the possible
  choices.

  - For now we're using one for a fixed ZPD steps ahead, another for a single
    forward pass among the loaded networks, and then a trainable network inside
    the loop.
  - The latter two are designed to use our overlap metrics in some way. If you
    use these, **the corresponding teacher directory must contain the relevant
    information (particularly saved neural networks) in the
    `distances` or `distances_multi` directories.**
  - The `prob_target` and `overlap_target` are values we can tune.
  - The `pretrained_model` key should represent the prefix of what model we
    use. If it is nothing (i.e., `""` or `null`) then by default we take the
    most recent model in `distances_multi`. We should be seeing stuff like this
    in there:
    ```
    -rw-rw-r-- 1 seita seita  17M Jun  6 08:32 r_2019-06-06-08-29_model_fold_0.pt
    -rw-rw-r-- 1 seita seita  17M Jun  6 08:35 r_2019-06-06-08-29_model_fold_1.pt
    -rw-rw-r-- 1 seita seita  17M Jun  6 08:40 r_2019-06-06-08-29_model_fold_2.pt
    -rw-rw-r-- 1 seita seita  17M Jun  6 08:43 r_2019-06-06-08-29_model_fold_3.pt
    -rw-rw-r-- 1 seita seita  17M Jun  6 08:46 r_2019-06-06-08-29_model_fold_4.pt
    -rw-rw-r-- 1 seita seita 6.7K Jun  6 08:29 r_2019-06-06-08-29_multiway.json
    -rw-rw-r-- 1 seita seita  55M Jun  6 08:48 r_2019-06-06-08-29_multiway_stats.p
    ```
    Thus, the `pretrained_model` in this case should be `r_2019-06-06-08-29`.
    There are multiple folds, which is nice because we can use ensembles for
    the forward pass. See [the distances package][1] for more details on how
    data is stored.

- Set `teacher_samples_uniform = true`! **This means we are NOT using progress
  scores**; that is, we just randomly sample from the teacher's replay buffer.
  It prevents confounding effects from other potential factors. When we want to
  test with progress scores, set this as `false` so that we go through the
  progress score computation.

- The `models` should be the name of the TEACHERS. They are lists to support
  ensembles, eventually. For now, use one.

- The `batch_size` is only for the progress network to compute progress scores
  by iterating through all the data loaded from some teacher. It's NOT the batch
  size representing the minibatch of samples that the student actually uses for
  training. For that, use the usual batch size parameter used in
  non-student/teacher training. That's usually 32 since DeepMind uses 32 by
  default.

- Use `num_snapshot_ahead` for testing hypotheses, and keep adjusting this.
  Unfortunately, it's manual, but unavoidable. We record the parameters after
  saving so we will not forget them. **If you set this to be -1, then this will
  be equivalent to using the best teacher as the ZPD each time.** That's
  actually a useful benchmark anyway.

- Keep `condense_freq=5` because that's the same as what we have for the overlap
  networks. This is only for fixed-aheads (it's ignored when we deal with the
  overlaps) and we further use `num_snapshots_ahead` on the resulting snapshots
  after the condense "filter" is applied.

- Keep `negative_correction=False`. I think it's one of the ways Allen wanted to
  make progress scores positive, but it's not the one we're using. We're instead
  going to just make all negative progress scores have the `progress_epsilon`
  parameter (see below). **TODO**

- Keep `progress_epsilon=0.000001`, we use to make negative progress scores this
  value, so they effectively won't be sampled. *I'm not sure why we need this,
  because we end up applying a softmax on the distribution of scores ...* Very
  confused. **TODO**

- Keep `temperature=1`. It's the temperature for a softmax, which I think we use
  for the teacher progress scores to turn them into a distribution. **TODO**

- Keep `dedup_iterations=0`, I don't actually know what this does; it's probably
  for ensembles of teachers. Only worry about this if `len(models) > 1`.
      **TODO**

- Use `init_wis_beat` for the initial beta in the weighted importance sampling
  procedure. These were around 0.5-ish for Prioritized Experience Replay, but we
  don't need the exact same values, and as long as they converge to 1 (a
  hard-coded value) we should be fine.

- The `replay_size` is the max size of how much data we load from a single
  teacher's snapshot. When we get to newer snapshots, we remove older ones, so
  that should save memory. BUT ... it also seems un-necessary to me because I'll
  be loading `num_teacher_samples` which handles that case? We shouldn't even
  get close to the `replay_size` ... maybe get rid of this?  Incidentally, the
  samples loaded form the teacher will be from the nearest "episode" (really,
  lifespan) and where we load samples before and after it, at an equal rate.
  This means we won't actually load corresponding to the episode cutoff points,
  but these are samples, so it's fine. (It's the same idea that I employ in the
  `distances` package.)

- The `blend` gives the teacher blend schedule. [The code is
  here](https://github.com/CannyLab/dqn/blob/42ea7d1fba74debd25f2d3fd57072f00735d32f1/dqn/teacher/teacher_centers.py#L178-L183).
  For now we might just keep it at 50-50, though after the last snapshot, it's
  ideal for the student to learn on its own ... so maybe keep `end: 0.5` and
  we'll manually change the other spot in the code. As of late March 2019,
  that's what I'm doing; I have a separate blend schedule that decays the
  teacher blend from 50 percent to 0 percent, but it seems like if I do that,
  the performance of the agent will dip ... particularly in Breakout. What to do
  about that?

- The `supervise_loss` or "imitation loss" finally gives details of the extra
  loss we add to ground Q-values appropriately. See the DQfD paper for more
  details. If we apply it to the learner, we also need to apply it to when we
  compute progress scores.

- On regularization: DQfD used L2 regularization (with lambda 0.00001) for
  their paper. But they didn't use it for their PDD-DQN baseline. It is only
  used for preventing overfitting to the expert data. So it is a judgment call
  as to whether we should include this or not, particularly becuase after we
  stop matching we only use self-generated data ... Todd Hester said it didn't
  have too much of an impact, but could be useful if we want to do lots of
  pre-training. **HUGE NOTE**: in PyTorch you can pass in `weight_decay` in the
  optimizer. That's what I'm doing for now.

[1]:https://github.com/CannyLab/distances
