*Note*: code used for the paper https://arxiv.org/abs/1910.12154

# Student-Teacher Training with (Double) Deep Q-Networks

Explores variations on the Deep Q-Learning algorithms, visualizes training and
trained agents, and imitates human learning methodologies.  Created by [Allen
Tang](http://allentang.me). Supports vanilla DQN, double DQN, and multi-step
DQN (but not currently in tandem with one-step DQN).

It also supports limited prioritization for XP replay for progress scores, but
this is not currently used.

When running student-teacher training, ensure that you are not using the same
teacher for multiple independent runs on the same machine. Some of our data
loading mechanisms depend on there being a known structure to the teacher's
directory, and we sometimes make new files on the fly.


## Installation

To install, set a Python 3 virtualenv. For reference, I'm using Python 3.5.2
(Python 3.6.7 also seems to work).

```bash
virtualenv --python=python3 seita-venvs/py3-whatever
```

and *activate it*. Then, *clone the repository*, change directory into it, and
do

`pip install -r requirements.txt`

I am using PyTorch 1.0.1.post2. There are some DDQN snapshots that were saved
using PyTorch 0.3.1, but later versions of PyTorch can load in earlier
snapshots.

Finally, run:

```bash
# Run from the root dqn directory
pip install -e .
```

so that this repository shows up in `pip freeze --local`, and the `develop`
option lets us work on this while seeing changes affected immediately (and
not having us re-install).

If you're using a computer with ROS, check that your `.bashrc` isn't
automatically sourcing ROS. Otherwise, the `PYTHONPATH` will point to the ROS
directory first, thus altering the priorities for Python packages. If you really
wish to support ROS, put the new virtualenv's `site-packages` directory before
the ROS ones in `PYTHONPATH`. But this is a bit hacky so I find the better
solution is just to comment out the entire ROS sourcing portion.


## Usage and Results

Look at `dqn/experiments` for usage and plotting.

# Miscellaneous

If you find this work relevant to yours, please consider citing:

```
@article{seita_zpd_2019,
    author = {Daniel Seita and Chen Tang and Roshan Rao and David Chan and Mandi Zhao and John Canny},
    title = {{ZPD Teaching Strategies for Deep Reinforcement Learning from Demonstrations}},
    journal={Deep Reinforcement Learning Workshop, NeurIPS},
    Year = {2019}
}
```
