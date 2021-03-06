import numpy as np
from matplotlib import pyplot as plt

from aligned import config


def plot_heatmap(mat,
                 y_tick_labels,
                 x_tick_labels,
                 label_interval: int = 10,
                 ):
    fig, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi * 4)
    plt.title('', fontsize=5)

    # heatmap
    print('Plotting heatmap...')
    ax.imshow(mat,
              aspect='equal',
              cmap=plt.get_cmap('jet'),
              interpolation='nearest')

    # x ticks
    x_tick_labels_spaced = []
    for i, l in enumerate(x_tick_labels):
        x_tick_labels_spaced.append(l if i % label_interval == 0 else '')

    num_cols = len(mat.T)
    ax.set_xticks(np.arange(num_cols))
    ax.xaxis.set_ticklabels(x_tick_labels_spaced, rotation=90, fontsize=1)

    # y ticks
    y_tick_labels_spaced = []
    for i, l in enumerate(y_tick_labels):
        y_tick_labels_spaced.append(l if i % label_interval == 0 else '')

    num_rows = len(mat)
    ax.set_yticks(np.arange(num_rows))
    ax.yaxis.set_ticklabels(y_tick_labels_spaced,  # no need to reverse (because no extent is set)
                            rotation=0, fontsize=1)

    # remove tick lines
    lines = (ax.xaxis.get_ticklines() +
             ax.yaxis.get_ticklines())
    plt.setp(lines, visible=False)
    plt.show()