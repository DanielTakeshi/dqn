import numpy as np
import scipy.stats


def boltzmann(x, T):
    return np.exp(-x / T) / np.exp(-x / T).sum()


def softmax_deepmind(x, T):
    return np.power(x, T) / np.power(x, T).sum()


def relative_perplexity(x):
    return np.power(2, scipy.stats.entropy(x, base=2)) / len(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
