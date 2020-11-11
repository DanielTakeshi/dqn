"""Help to manage different computer and game usage.
"""
import platform
import socket
import os
import warnings

# NUMBER ONE ITEM TO BE AWARE OF!!!
COMPUTER = platform.node()
assert platform.node() == socket.gethostname()

# ------------------------------------------------------------------------------
# Top-level directory where we save trained agent snapshots and trained models,
# plus datasets of images for training CNNs. Note on CPUs: my machine and Biye's
# both use 4 CPUs while the Tritons use 6 CPUs.
# ------------------------------------------------------------------------------

# Deprecation warning for computer which are set in the common_config file.
# Update October 2019: we are not using this but just leaving for posterity's sake.
common_configs = {
    'takeshi': '/media/daniel/bigdata/mt-dqn',
    'BID-Biye-v2': '/data/mt-dqn/',
    'stout': '/big/home/seita/mt-dqn',
    'jfc-devbox': '/raid/data/mt-dqn/',
    'autolab-titan-box': '/nfs/diskstation/seita/mt-dqn',
    'triton2': '/nfs/diskstation/seita/mt-dqn',
    'triton3': '/nfs/diskstation/seita/mt-dqn',
    'triton4': '/nfs/diskstation/seita/mt-dqn',
    'jensen': '/raid/seita/mt-dqn'}

if os.path.exists(os.environ['DQN_DATA_DIR']):
    DATA_HEAD = os.environ['DQN_DATA_DIR']
elif os.environ['DQN_DATA_DIR']:
    raise ValueError('DQN_DATA_DIR environment variable set, but that \
                      directory ({}) does not exist'.format(os.environ['DQN_DATA_DIR'])
    )
elif COMPUTER in common_configs:
    warnings.warn('Setting DATA_HEAD in the common_config.py is deprecated, \
        please set this using the DQN_DATA_DIR environment variable instead.')
    DATA_HEAD = common_configs[COMPUTER]
else:
    raise ValueError('Please configure the DATA_HEAD for this computer ({}) by \
        setting the DQN_DATA_DIR environment variable'.format(COMPUTER))
assert os.path.exists(DATA_HEAD)

# ------------------------------------------------------------------------------
# teacher: anything that doesn't involve training a student, just view as normal
#          training
# student: anything that involves student-teacher interaction, save _student_
#          models
# ------------------------------------------------------------------------------

SNAPS_TEACHER = os.path.join(DATA_HEAD, 'teachers/')
SNAPS_STUDENT = os.path.join(DATA_HEAD, 'students/')
