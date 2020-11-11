import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.colors as colors
import seaborn as sns
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
sns.set_style("darkgrid")
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def millions(x, pos):
    return '%1.1fM' % (x*1e-6)


def ints(x, pos):
    return int(x)


million_formatter = FuncFormatter(millions)
int_formatter = FuncFormatter(ints)


def plot_state(states, next_states):
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    num_frams = states.shape[0]
    frame_size = states.shape[1]
    canvas = np.zeros((frame_size * 2, frame_size * num_frams))
    for j in range(num_frams):
        canvas[0:frame_size, j*frame_size:(j+1)*frame_size] = states[j]
        canvas[frame_size:2*frame_size, j*frame_size:(j+1)*frame_size] = \
            next_states[j]
        canvas[:, j*frame_size] = 0
    canvas[frame_size] = 0
    ax.axis('off')
    ax.imshow(canvas, cmap='gray')
    ax.set_title('Transition')
    plt.show()


def generate_transition(transition):
    state = transition.state[0][-1]
    next_state = transition.next_state[0][-1]
    frame_size = state.shape[1]
    canvas = np.zeros((frame_size, frame_size * 2))
    canvas[0:frame_size, 0:frame_size] = state
    canvas[0:frame_size, frame_size:(2*frame_size)] = next_state
    canvas[:, frame_size] = 0
    return canvas


def generate_top_transitions_grid(n, sorted_index, replay, num_steps, top=True):
    _transitions = []
    for i in range(n):
        if top:
            idx = i
        else:
            idx = (i + 1) * -1
        transition = replay.get_transitions(
            range(sorted_index[idx], sorted_index[idx] + 1),
            num_steps=num_steps, force=True)
        _transitions.append(generate_transition(transition))

    # _transitions = np.concatenate(_transitions)
    # _transitions = np.expand_dims(_transitions, axis=1)
    return np.stack(_transitions)


def _plot_transition(transition, title="Transition", ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 2))
    assert len(transition) == 1
    canvas = generate_transition(transition)
    ax.axis('off')
    ax.imshow(canvas, cmap='gray')
    ax.set_title(title)


def _plot_top_transitions(snapshot_num, replay, idx, values, top=True, n=3):
    f, ax = plt.subplots(n, n, figsize=(4 * n, 2 * n))
    plt.suptitle("Sample Transitions with Most Extreme Progress Score at "
                 "Snapshot {0}".format(snapshot_num))
    for i in range(0, n):
        for j in range(0, n):
            k = i*n + j
            title = "Bottom #{0} transition with score {1:.2f}".format(
                abs(k), values[idx[k]])
            if top:
                k = (k + 1) * -1
                title = "Top #{0} transition with score {1:.2f}".format(
                    abs(k), values[idx[k]])
            transition = replay.get_samples(range(idx[k], idx[k]+1),
                                            num_steps=1)
            _plot_transition(transition=transition, title=title, ax=ax[i, j])
    plt.show()


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_hist(x, title, file_name=None):
    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.hist(x, bins=30, range=(np.percentile(x, 3), np.percentile(x, 97)))
    ax.set_title(
        "Distribution of {0} \n (Display middle 95%, mean {1:.2f}, std {2:.2f})"
        .format(title, np.mean(x), np.std(x)))
    if file_name is None:
        plt.show()
    else:
        f.savefig(file_name)


def plot_tsne(sample_index, replay):
    fig = Figure(figsize=(8, 8))
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    pca = PCA(n_components=50)
    embeddings = pca.fit_transform(replay.info["activations"][sample_index])
    tsne_embeddings = TSNE(n_components=2, perplexity=40).fit_transform(
        embeddings)
    s = ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 0.3,
                   c=replay.info["progress_scores"][sample_index],
                   norm=MidpointNormalize(midpoint=0), cmap='seismic')
    fig.colorbar(s)
    canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    return buf
