
import matplotlib.pyplot as plt
import numpy as np


def plot_atten_align(att, chain, score, title='Encoder', y_label='Binding Probability Rank', cmap='viridis', figsize=(15,15)):
    weight_map = att
    seq_units = chain
    scores = score
    max_len = max([len(unit) for unit in seq_units]) + 1
    seq_units = [f'-{unit:-<{max_len-1}}' for unit in seq_units]
    # max_len = np.sum(weight_map > 0, axis=-1).max()
    weight_map = weight_map[:, :max_len]
    weight_mean = np.mean(weight_map, axis=0)[None, :]
    weight_map = np.concatenate([weight_mean, weight_map])

    fig = plt.figure(figsize=figsize)
    bx = fig.subplots()

    bx.imshow(weight_map, cmap=cmap)
    for x, seq in enumerate(seq_units):
        for y, unit in enumerate(seq):
            bx.text(y, x+1, unit, ha="center", va="center", color="w")
    for i, w in enumerate(weight_mean[0]):
        bx.text(i, 0, f'{w/np.max(weight_mean):.2f}', ha="center", va="center", color="w")
    bx.set_yticks(np.arange(0, len(seq_units)+1), ['MEAN', *[f'{i:.2f}'for i in scores]])
    bx.set_xticks(np.arange(0, max_len, 1), ['-']+np.arange(1, max_len, 1).tolist())
    bx.set_xlabel('Sequence')
    bx.set_ylabel(y_label)
    bx.set_title(title)
    return fig