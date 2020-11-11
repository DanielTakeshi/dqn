import torch
import os
import logging


class Agent(object):
    def __init__(self, net, gpu_params, log_params, policy, tag):
        self._net = net
        self._gpu_params = gpu_params
        self._log_params = log_params
        self.logger = logging.getLogger("agent" + tag)
        self._policy = policy

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def get_net(self):
        return self._net

    def get_policy_epsilon(self, steps):
        if self._policy:
            return self._policy.get_epsilon(steps)
        else:
            return None

    def save_model(self, snapshot_num):
        """Called in the subclasses to save same way as in PyTorch.
        Note: subclasses Agent -> DQNAgent -> DQNTrainAgent.
        """
        torch.save(self._net.state_dict(), os.path.join(
            self._log_params["dir_snapshots"],
            "snapshot_{}.pth.tar".format(str(snapshot_num).zfill(4))))
        self.logger.info(
            "Snapshot {} of the Q network has been saved on disk."
            .format(snapshot_num))

    def save_model_newpath(self, path):
        """Gives flexibility for saving to a certain path.
        """
        torch.save(self._net.state_dict(), path)
