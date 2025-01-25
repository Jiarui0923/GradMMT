from .thresholds import Thresholds
from tabulate import tabulate
import numpy as np
import pandas as pd


class SequencePrototype(object):
    
    def __init__(self, sequences, activations, scores, ids=None,
                 backbone_threshold_func=Thresholds.percentile(0.15),
                 sample_threshold_func=Thresholds.percentile(0.15),):
        self.max_seq_len = max([len(unit) for unit in sequences])
        self.backbone_mask = self._build_mask(activations, self.max_seq_len, backbone_threshold_func)
        sequences = np.stack([np.array(list(unit + ('-'*(self.max_seq_len-len(unit))))) for unit in sequences])
        self.backbone = self._build_backbone(sequences, self.backbone_mask)
        self.base_score = self._build_base_score(scores, sequences, self.backbone)
        self.base_act = self._build_base_act(self.backbone_mask, activations, self.max_seq_len)
        self.blocks = self._build_cluster_offset(sequences, activations, scores, sample_threshold_func, backbone_threshold_func)
        self.filters = self.seperate(scores, activations)
        self.shifts = (self.calculate_backbone_shift(activations[self.filters[0][0] | self.filters[0][1]]),
                       self.calculate_backbone_shift(activations[self.filters[1][0] | self.filters[1][1]]))
        if self.shifts[0] > 0 and self.shifts[1] < 0: self.shift_symbol = '+ => +'
        elif self.shifts[0] < 0 and self.shifts[1] > 0: self.shift_symbol = '+ => -'
        else: self.shift_symbol = 'ERR'
        
        if ids is None: self.ids = sequences
        else: self.ids = ids
        
    
    def __len__(self): return len(self.blocks)
    def __repr__(self): return f'< Prototype: {self.backbone} (+{len(self)}) [{self.shift_symbol}: {self.base_score:.4f}] >'
    def _repr_markdown_(self):
        print(self.block_table)
        return f'`{self.backbone}` (+{len(self)}) [`{self.shift_symbol}`: {self.base_score:.4f}]'
    
    @property
    def df(self):
        offset, filters, ids = self.blocks, self.filters, self.ids
        _df = None
        for i, i_s in zip([0,1], ['+', '-']):
            for j, j_s in zip([0,1], ['+', '-']):
                section = offset[filters[i][j]].copy()
                if len(section) > 0:
                    section['shift'] = f'{i_s} => {j_s}'
                    section['ids'] = ids[filters[i][j]]
                    section['backbone'] = self.backbone
                    section['base_activation'] = np.mean(self.base_act)
                    section['base_score'] = self.base_score
                    if _df is None: _df = section
                    else: _df = pd.concat([_df, section])
        return _df
        
    def _build_mask(self, weight_map, max_length, act_threshold_func=Thresholds.percentile(0.15)):
        weight_map = weight_map[:, :max_length]
        # threshold = act_threshold_func(weight_map)
        # mask = (weight_map > threshold)
        # char_mask = (np.mean(mask, axis=0) > 0)
        threshold = act_threshold_func(np.mean(weight_map, axis=0))
        char_mask = (np.mean(weight_map, axis=0) > threshold)
        return char_mask
    
    def _build_backbone(self, sequences, char_mask):
        backbones = sequences
        chars, rank = np.unique(backbones.T[char_mask], axis=1, return_counts=True)
        backbone = chars.T[np.argmax(rank)]
        blank = np.array(list('-'*char_mask.shape[-1]))
        blank[char_mask] = backbone
        backbone = ''.join(blank)
        return backbone
    
    def _build_base_score(self, scores, sequences, backbone):
        backbones = sequences
        blank = np.array(list(backbone))
        dists_percent = 2**np.sum(backbones == blank, axis=1)
        dists_percent = dists_percent / np.sum(dists_percent)
        baseline_score = np.sum(scores * dists_percent)
        # baseline_score = np.median(scores)
        return baseline_score
    
    def _build_base_act(self, char_mask, activations, max_length):
        return np.mean(activations[:, :max_length]*char_mask, axis=0)
    
    def get_backbone_act(self, activations):
        return np.mean(activations[:, :self.max_seq_len] * self.backbone_mask, axis=1)
    
    def seperate(self, scores, activations):
        _score_filter = scores > self.base_score
        _act_filter = self.get_backbone_act(activations) > np.mean(self.base_act)
        return ((_score_filter&_act_filter, _score_filter&~_act_filter),
                (~_score_filter&_act_filter, ~_score_filter&~_act_filter))
        
    def _build_cluster_offset(self, chains, weights, scores,
                              threhold_func=Thresholds.percentile(0.01),
                              backbone_threhold_func=Thresholds.percentile(0.15),):
        weights = weights[:, :self.max_seq_len]
        cols = np.stack([np.array(list(unit)) for unit in chains]).T
        vars = np.array([len(np.unique(col)) for col in cols])
        var_mask = vars > 1

        data = []
        for chain, weight, score in zip(chains, weights, scores):
            threhold = threhold_func(weight)
            seq = np.array(list(chain))
            _mask = (weight > threhold)[:len(self.backbone_mask)]
            _mask = _mask & (~self.backbone_mask)
            mask = _mask & var_mask
            offet = np.array(list('-'*len(seq)))
            offet[mask] = seq[mask]
            _backbone_update_mask = (seq != np.array(list(self.backbone))) & (self.backbone_mask)
            offet[_backbone_update_mask] = seq[_backbone_update_mask]
            
            _migrate_backbone = np.array(list('-'*len(seq)))
            _mask = (weight > backbone_threhold_func(weight))[:len(self.backbone_mask)]
            _migrate_backbone[_mask] = seq[_mask]

            weight_backbone_mean = np.mean(weight * self.backbone_mask)
            data.append({'offset':''.join(offet),
                        'backbone_act':weight_backbone_mean,
                        'backbone_migrate':''.join(_migrate_backbone),
                        'score':score,
                        'chain':''.join(chain)})
        return pd.DataFrame(data)
    
    def calculate_backbone_shift(self, activations):
        activations = self.get_backbone_act(activations)
        return np.mean(activations- np.mean(self.base_act))
    
    @property
    def block_table(self):
        offset, filters = self.blocks, self.filters
        lines = []
        for i, i_s in zip([0,1], ['+', '-']):
            for j, j_s in zip([0,1], ['+', '-']):
                section = offset[filters[i][j]].copy()
                if len(section) > 0:
                    section['shift'] = f'{i_s} => {j_s}'
                    lines += section.values.tolist()
            if i == 0:
                lines += [[self.backbone,
                        np.mean(self.base_act),
                        self.backbone,
                        self.base_score,
                        self.backbone,
                        'BACKBONE']]
        return tabulate(lines, floatfmt='.4f',
                        tablefmt='fancy_outline',
                        headers=['Offset', 'Act', 'Migrate', 'Score', 'Chain', 'Shift'])

    @property
    def block_table_df(self):
        offset, filters = self.blocks, self.filters
        lines = []
        for i, i_s in zip([0,1], ['+', '-']):
            for j, j_s in zip([0,1], ['+', '-']):
                section = offset[filters[i][j]].copy()
                if len(section) > 0:
                    section['shift'] = f'{i_s} => {j_s}'
                    lines += section.values.tolist()
            if i == 0:
                lines += [[self.backbone,
                        np.mean(self.base_act),
                        self.backbone,
                        self.base_score,
                        self.backbone,
                        'BACKBONE']]
        return pd.DataFrame(lines, columns=['Offset', 'Act', 'Migrate', 'Score', 'Chain', 'Shift'])
        

from hashlib import md5
class DataFrame2Prototype(object):
    threshold_backbone = Thresholds.percentile(0.5)
    threshold_sample = Thresholds.percentile(0.9)
    hash_method = md5
    
    @classmethod
    def _hash(cls, df, index_cols=['chain_alpha', 'chain_beta', 'chain_epitope']):
        return df[index_cols].apply(lambda x : cls.hash_method(''.join([' ' if i is None else str(i) for i in x.values]).encode()).hexdigest(),
                             axis=1)
    @classmethod
    def _uni(cls, df, chain_col, weight_col, score_col='score',
             index_cols=['chain_alpha', 'chain_beta', 'chain_epitope'],
             remove_start_symbol=True):
        return SequencePrototype(df[chain_col].values,
                                 np.stack(df[weight_col])[:, 1:] if remove_start_symbol else np.stack(df[weight_col]),
                                 df[score_col].values,
                                 cls._hash(df, index_cols),
                                 cls.threshold_backbone,
                                 cls.threshold_sample)
    @classmethod
    def beta_decode(cls, df):
        return cls._uni(df, chain_col='chain_beta', weight_col='decode_beta',
                        index_cols=['chain_alpha', 'chain_beta', 'chain_epitope'])
    
    @classmethod
    def alpha_decode(cls, df):
        return cls._uni(df, chain_col='chain_alpha', weight_col='decode_alpha',
                        index_cols=['chain_alpha', 'chain_beta', 'chain_epitope'])
        
    @classmethod
    def epitope_decode(cls, df):
        return cls._uni(df, chain_col='chain_epitope', weight_col='decode_epitope',
                        index_cols=['chain_alpha', 'chain_beta', 'chain_epitope'])
    

def table_confusion_matrix_backbone(analyzer, conditions = {}, trans_func=DataFrame2Prototype.beta_decode,
                                    labels=['TP', 'TN', 'FP', 'FN']):
    secs = analyzer.confusion_matrix_sections.matrix
    tables = {}
    for sec, label in zip(secs, labels):
        sec = analyzer.fetch(sec, conditions)
        if len(sec) > 0:
            tables[label] = trans_func(analyzer.fetch(sec, conditions)).block_table_df
    return tables