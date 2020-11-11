import torch.optim as optim
from dqn.utils.schedules import PiecewiseSchedule
import logging


class Optimizer:

    def __init__(self, net, opt_params, max_num_steps, train_freq_per_step):
        """
        `Optimizer` object wraps on top of PyTorch `optim` but with more
        freedom to control learning rates decay and etc. For the sake of
        clarity, just keep the `multiplier` at 1 so the learning rate is just
        the learning rate directly... `decay_lr` is the final value, but we can
        just make it equal to the normal `lr`.

        See: https://pytorch.org/docs/stable/_modules/torch/optim/adam.html for
        arguments to Adam optimizer, from `self.params['params']`.

        :param net: neural network that optimizer will act on.
        :param opt_params: hyper-parameters for optimizer.
        :param max_num_steps: maximum number of steps to do in the environment.
        :param train_freq_per_step: number of steps between two training
        iterations.
        """
        self.params = opt_params
        self.params["params"]["lr"] *= self.params["multiplier"]
        self.params["decay_lr"] *= self.params["multiplier"]
        if self.params["name"] == "Adam":
            self.opt = optim.Adam(net.parameters(), **self.params["params"])
        else:
            self.opt = optim.RMSprop(net.parameters(), **self.params["params"])
        self.clipping = self.params["clipping"]
        _num_iterations = max_num_steps // train_freq_per_step
        self.lr_schedule = PiecewiseSchedule([
            (0,                    self.params["params"]["lr"]),
            (_num_iterations / 10, self.params["params"]["lr"]),
            (_num_iterations / 2,  self.params["decay_lr"]),
        ], outside_value=self.params["decay_lr"])
        # D: e.g., LR at 0.00025, stays for 10% of iters, then decays to 0.0001.
        logger = logging.getLogger("optimizer") 
        logger.debug("Just defined LR schedule.")
        logger.debug("  LR _num_iterations: {}".format( _num_iterations))
        logger.debug("  LR start: {}".format(self.params["params"]["lr"]))
        logger.debug("  LR decay: {}".format(self.params["decay_lr"]))


    def get_lr(self, steps):
        """
        Given the number of steps, return the current learning rates.
        :param steps: Number of training steps that t
        :return: learning rates
        """
        return self.lr_schedule.value(steps)


    def get_opt(self, steps):
        """
        Given the number of steps, return an optimizer with adjusted learning
        rates.
        :param steps: Number of training steps that t
        :return: An optimizer
        """
        for param_group in self.opt.param_groups:
            param_group['lr'] = self.lr_schedule.value(steps)
        return self.opt
