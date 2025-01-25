
from sklearn.metrics import roc_auc_score
import numpy as np


class ConfusionMatrixSections(object):
    def __init__(self, tp, tn, fp, fn):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
    @property
    def quant(self):
        return len(self.tp), len(self.tn), len(self.fp), len(self.fn)
    @property
    def matrix(self):
        return self.tp, self.tn, self.fp, self.fn
    @property
    def recall(self):
        tp, tn, fp, fn = self.quant
        return tp / (tp + fn)
    @property
    def precision(self):
        tp, tn, fp, fn = self.quant
        return tp / (tp + fp)
    @property
    def f1(self):
        tp, tn, fp, fn = self.quant
        return (2 * tp) / (2 * tp + fp + fn)
    @property
    def accuracy(self):
        tp, tn, fp, fn = self.quant
        return (tp + tn) / (tp + tn + fp + fn)
    @property
    def report(self):
        return {'recall': self.recall, 'precision': self.precision, 'f1': self.f1, 'accuracy': self.accuracy}


class AnalyzerCollector:
    @staticmethod
    def beta(x): return x[x.chain_beta.notna()]
    @staticmethod
    def epitope(x): return x[x.chain_epitope.notna()]
    @staticmethod
    def simple(x): return x

class RolloutAnalyzer(object):

    def __init__(self, case, collector=AnalyzerCollector.simple):
        self.case = case
        self.collector = collector

    @property
    def df(self):
        return self.collector(self.case.df)

    def fetch(self, df, conditions={}):
        filter = np.ones(len(df), dtype=np.bool_)
        if 'epitope' in conditions: filter &= (df.chain_epitope == conditions['epitope'])
        if 'beta' in conditions: filter &= (df.chain_beta == conditions['beta'])
        if 'alpha' in conditions: filter &= (df.chain_alpha == conditions['alpha'])
        return df[filter]
    
    @property
    def auc_roc(self):
        return roc_auc_score(self.df.binder, self.df.score)

    @property
    def confusion_matrix_sections(self):
        df = self.collector(self.case.df)
        pred_scores = np.argsort(np.argsort(df.score))
        neg_index = np.argsort(pred_scores)[:(df.binder == 0).sum()].values
        pos_index = np.argsort(pred_scores)[(df.binder == 0).sum():].values
        pos_df = df.iloc[pos_index]
        neg_df = df.iloc[neg_index]
        return ConfusionMatrixSections(pos_df[pos_df.binder == 1],
                                       pos_df[pos_df.binder == 0],
                                       neg_df[neg_df.binder == 0],
                                       neg_df[neg_df.binder == 1])



    