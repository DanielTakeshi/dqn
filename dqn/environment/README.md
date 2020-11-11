# Environment Wrapping


## Frames and Wrappers

The wrapping is similar to OpenAI's code:

https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

Similarities: we use the set of wrappers for no-ops, max and skip, episodic,
fire, warp, clip reward, and frame stack, and both the max and skip and frame
stack get 4 as the argument, so that's good. We also, like OpenAI, do NOT use
the `ScaledFloatFrame` wrapper. We want our types as `np.uint8`, not floats!!
OpenAI does the scaling here:

- https://github.com/openai/baselines/blob/master/baselines/common/models.py
- https://github.com/openai/baselines/pull/632

We do the scaling in the forward pass of our PyTorch models:

- https://github.com/CannyLab/dqn/blob/master/dqn/models.py

Basically, OpenAI and ourselves agree on these wrappers:

```
NoopResetEnv(env, noop_max=30)
MaxAndSkipEnv(env, skip=skip_frame)
EpisodicLifeEnv(env)
FireResetEnv(env)
WarpFrame(env)
FrameStack(env, frame_stack)
ClipRewardEnv(env)
```

Main difference: Allen introduced the `ImageToPyTorch` wrapper and put it after
`WarpFrame` and before `FrameStack`, because PyTorch uses different ordering of
dimensions as TensorFlow. :(

*In addition*, Allen modified `WarpFrame` so that it returns TWO things: the
warped frames (with correct RGB to grayscale and resizing) AND the original
frames! This tuple is then passed through the subsequent wrappers which can deal
with both observations. Rationale: we also want the original, non-resized color
images. *I don't use this but since I don't want to break existing code, we'll
leave it here.*


## Monitors and Rewards

OpenAI uses monitors, see:

- https://github.com/openai/baselines/issues/667
- https://github.com/openai/baselines/blob/master/baselines/common/cmd_util.py
  (look at `make_env` method, that's used for DQN with Atari).

Note: I think the terminology in pzhokhov's comment is a bit ambiguous. They
actually apply `NoopResetEnv` and `MaxAndSkipEnv` before `bench.Monitor`, but
indeed, `bench.Monitor` is then used.

Allen did not originally have that, which is causing problems, see
https://github.com/CannyLab/dqn/issues/6. So, I added it after the `make_atari`
method, again following baselines.

There is still other changes to supporting code that we must make, though, about
detecting when episodes end. See:

- https://github.com/CannyLab/baselines-sandbox/issues/9
- https://github.com/openai/baselines/issues/42

Look at `EpisodicLifeEnv`:

https://github.com/openai/baselines/blob/57e05eb420f9a20fa8cd7ee7580b1a8874b4a323/baselines/common/atari_wrappers.py#L59-L93

- In particular, the `self.was_real_done` will be done ONLY WHEN ALL LIVES ARE
  ZERO (or when we're actually at a 'done' in games that have no lives, such as
  Pong). It's just that we RETURN `done` if the lives has decreased, hence he
  whole `if lives < self.lives` thing.
  
- `self.lives` is updated with the lives, so if we lost a life, then `lives <
  self.lives`, should be `lives = self.lives=1` but might be the rare case where
  a few steps automatically loses many lives? I doubt it but they have that just
  to be safe.

- If `lives=0` then we ignore the if condition, but that's fine, we have `obs,
  rew, done, info` from the environment stepping, and if `done=True`, that is
  the ACTUAL done we need to fully reset the environment.

- We reset environments to start with, and this will automatically populate
  `self.lives` with the true number of lives. Good.

- AH, we CAN call `env.reset()` even if we're at a `done=True` but
  `self.was_real_done=False` case. The reset will do no-op steps.

What does this imply for our code?

- Actually, Allen *did* use `EpisodicLifeEnv` --- it's just that he detects
  when episodes are done by whether `done` from `env.step()` is True, but that's
  not the correct way to detect it. He does correctly reset, which is why the
  monitor will be able to track the correct, un-clipped scores.

- Following [OpenAI baselines deepq code][1], we *still* need to call
  `obs.reset()` if `done` from the `env.step()` is True, because by doing so
  that will call the `EpisodicLifeEnv`'s reset method, which steps forward with
  no-ops if we still have lives left. (In baselines, for other code it's a bit
  trickier due to parallel envs which automatically reset, but their DQN code is
  clear, if we see `done=True` from `env.step` we must reset.) It's just that
  when we *record* all this in episodes, we have to keep appending to our
  current episode if we still have lives.

- Making it challenging somewhat, we do the stepping and resetting in different
  methods. I tried to get it to work, but just went into too many difficulties.
  I thus decided to simply keep it the way it is, except for reporting results
  we'll have to look at the output from `monitor.csv`, NOT the reward files we
  saved.

- In other words, each 'episode' in the replay buffer needs to be re-interpreted
  as a 'single lifespan', which is only a true full episode for games like Pong
  without lives. With debug messages, we can make this slightly more explicit.

[1]:https://github.com/openai/baselines/blob/master/baselines/deepq/deepq.py
