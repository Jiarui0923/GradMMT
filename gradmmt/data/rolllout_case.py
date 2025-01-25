import os
import pandas as pd
import numpy as np

class RolloutCase(object):
    _full_frame_path = 'model_output_full.json'
    def __init__(self, path):
        self._path = path
        if not os.path.exists(self.path(self._full_frame_path)):
            self.df = self._merge_docs(to_file=self._full_frame_path)
        else: self.df = pd.read_json(self.path(self._full_frame_path))
        
        cols = ['embed_alpha', 'embed_beta', 'embed_epitope',
        'decode_alpha', 'decode_beta', 'decode_epitope',
        'encode_alpha', 'encode_beta', 'encode_epitope']
        for col in cols:
            self.df[col] = self.df[col].apply(np.array)
        self.df['score'] = self.df['score'].astype(float)
    def path(self, *args):
        return os.path.join(self._path, *args)
    def _merge_docs(self, distributed_folder='model_output', to_file='model_output_full.json'):
        for root, _, files in os.walk(self.path(distributed_folder)):
            df = pd.DataFrame([])
            for file in files:
                _df_frag = pd.read_json(os.path.join(root, file))
                df = pd.concat([df, _df_frag])
        df.reset_index(drop=True, inplace=True)
        df.to_json(self.path(to_file))
        return df