import numpy as np
from .att_vis import plot_atten_align

def plot_confusion_matrix_activation(analyzer, conditions = {}, value_col='decode_beta', chain_col='chain_beta',
                                     titles=['True Positve', 'True Negative',
                                             'False Positve', 'False Negative'],
                                     cmaps=['viridis', 'viridis', 'viridis', 'viridis'], figsize=(15,15)):
    secs = analyzer.confusion_matrix_sections.matrix
    figs = {}
    for title, sec, cmap in zip(titles, secs, cmaps):
        _df = analyzer.fetch(sec, conditions)
        if len(_df) > 0:
            scores = np.argsort(np.argsort(_df.score.values))
            figs[title] = plot_atten_align(np.stack(_df[value_col].values),  _df[chain_col], scores, title=title, cmap=cmap, figsize=figsize)
    return figs