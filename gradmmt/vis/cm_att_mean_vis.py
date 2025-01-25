import numpy as np
import matplotlib.pyplot as plt

def cal_act_mean(att, chain):
    max_len = max([len(unit) for unit in chain]) + 1
    weight_map = att[:, :max_len]
    weight_mean = np.mean(weight_map, axis=0)
    return weight_mean/np.max(weight_mean)


def plot_confusion_matrix_activation_mean(analyzer, conditions = {}, value_col='decode_beta', chain_col='chain_beta', labels=['TP', 'TN', 'FP', 'FN'], title='Mean Activtation'):
    secs = analyzer.confusion_matrix_sections.matrix
    fig = plt.figure()
    ax = fig.subplots()
    _max_len = 0
    line_styles = ['-', '-', '--', '--']
    for sec, label, linestyle in zip(secs, labels, line_styles):
        sec = analyzer.fetch(sec, conditions)
        if len(sec) > 0:
            m = cal_act_mean(np.stack(sec[value_col]), sec[chain_col])
            if _max_len < len(m): _max_len = len(m)
            ax.plot(m, label=label, linestyle=linestyle)
    ax.grid(linestyle=':')
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Sequence')
    ax.set_ylabel('Activtion')
    if len(analyzer.fetch(analyzer.df, conditions=conditions)[chain_col].unique()) > 1:
        ax.set_xticks(np.arange(_max_len), np.arange(1, 1+_max_len))
    else:
        ax.set_xticks(np.arange(_max_len), [f'{c}\n{i}' for i, c in enumerate(['-']+list(analyzer.fetch(analyzer.df, conditions=conditions)[chain_col].values[0]))])
    return fig