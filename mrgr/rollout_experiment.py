import docflow as doc
from . import models
from . import tokenizers
from . import interprets
from . import data
from tqdm import tqdm
import json
import torch
import random
import string
import os
import pandas as pd
import numpy  as np
from torch import nn
from torch.nn import functional as F
from datetime import datetime

class DocumentRecorder(object):
    def __init__(self, title='Experiment'):
        self.title = title
        self.records = {
            'config': {}, 'track': {}, 'dataset':{}, 'experiment':{},
        }
    @property
    def _indentifier(self): return ''.join(random.sample(string.digits+string.ascii_uppercase, k=8))
    def add_config(self, section='', configs={}):
        self.records['config'][section] = configs
    def doc_config(self):
        return doc.Document(
            doc.Title('Configuration', level=2),
            *[
                doc.Document(
                    doc.Title(section, level=3),
                    doc.Sequence(content),
                ) for section, content in self.records['config'].items()
            ]
        )
    def add_track(self, tracks):
        self.records['track'] = tracks
    def doc_track(self):
        return doc.Document(
            doc.Title('Tracks', level=2),
            *[
                doc.Document(
                    doc.Title(f'Track-{index}', level=3),
                    doc.Sequence(track),
                ) for index, track in enumerate(self.records['track'])
            ]
        )
    def add_exp(self, outcomes):
        self.records['experiment'] = outcomes
    def doc_exp(self):
        return doc.Document(
            doc.Title('Experiments', level=2),
            doc.Sequence(self.records['experiment'])
        )
    def add_dataset(self, section='', configs={}):
        self.records['dataset'][section] = configs
    def doc_dataset(self):
        return doc.Document(
            doc.Title('Dataset', level=2),
            *[
                doc.Document(
                    doc.Title(section, level=3),
                    doc.Sequence(content),
                ) for section, content in self.records['dataset'].items()
            ]
        )
    def _repr_markdown_(self): return self.document._repr_markdown_()
    @property
    def document(self):
        return doc.Document(
            doc.Title(self.title, level=1),
            doc.DateTimeStamp('%d-%m-%Y %H:%M:%S'),
            doc.Text('\n'),
            doc.IdenticalBadge(),
            doc.Badge('ID', self._indentifier, color='0080c9'),
            self.doc_config(),
            self.doc_track(),
            self.doc_exp()
        )
        
class Experiment(object):
    _recorder = DocumentRecorder
    def __init__(self, name='Experiment', storage_path='./', config=None):
        self.recorder = self._recorder(title=name)
        _subfolder = datetime.now().strftime('%y%m%d%H%M%S')
        storage_path = f'{storage_path}/{_subfolder}'
        os.makedirs(storage_path, exist_ok=True)
        self.storage_path = storage_path
        if config is not None: self.load(config)
    def path(self, *args):
        return os.path.join(self.storage_path, *args)
    
    def load(self, config):
        self.config_dict = config
        self.setup(**config)
        
    def setup(self, model_config, tracks, dataset_path, mhc=False, max_length=50,
              device='cpu', checkpoint_path='./', random_seed=0, batch_size=512,
              shuffle=False, special_chars=[2,3,4], discard_ratio=0.9):
        self.setup_general(random_seed=random_seed)
        self.setup_model(model_config=model_config, mhc=mhc,
                         max_length=max_length, device=device,
                         checkpoint_path=checkpoint_path)
        self.setup_interpret(tracks=tracks, discard_ratio=discard_ratio)
        self.setup_data(path=dataset_path, batch_size=batch_size, shuffle=shuffle, device=device)
        self.setup_runtime(pad_to=max_length, special_chars=special_chars)
    def setup_general(self, random_seed=0):
        torch.manual_seed(random_seed)
        self.recorder.add_config('General', {
            'random_seed': random_seed,
            'storage_path': self.storage_path
        })
    def setup_model(self, model_config, mhc=False, max_length=50, device='cpu', checkpoint_path='./'):
        aa_tokenizer = tokenizers.amino_acids()
        mhc_tokenizer = tokenizers.mhc(enable=mhc)
        aa_vocab_size=len(aa_tokenizer._tokenizer.get_vocab())
        mhc_vocab_size=len(mhc_tokenizer._tokenizer.get_vocab())
        self.recorder.add_config('Tokenizer',
                                 {'mhc_token': mhc,
                                  'amino_acid_vocab_size': aa_vocab_size,
                                  'mhc_vocab_size': mhc_vocab_size,
                                  'max_length': max_length})
        if isinstance(model_config, str):
            with open(model_config, "r") as read_file: model_config = json.load(read_file)
        self.recorder.add_config('Model', model_config)
        tulip = models.Tulip(config=model_config, max_length=max_length,
                             aa_vocab_size=aa_vocab_size, mhc_vocab_size=mhc_vocab_size,
                             pad_id=aa_tokenizer.pad_token_id)
        checkpoint = torch.load(checkpoint_path)
        tulip.load_state_dict(checkpoint)
        tulip = tulip.to(device)
        self.recorder.add_config('Model Runtime',
                                 {'device': str(device),
                                  'checkpoint_path': checkpoint_path})
        self.tulip = tulip
        self.aa_tokenizer = aa_tokenizer
        self.mhc_tokenizer = mhc_tokenizer
    def setup_interpret(self, tracks, discard_ratio=0.9):
        tulip_track = interprets.TulipMultiTrack(self.tulip, tracks=tracks)
        self.recorder.add_track(tracks)
        self.tulip_track = tulip_track
        self.discard_ratio = discard_ratio
        self.recorder.add_config('Rollout',
                                 {'discard_ratio': discard_ratio})
    def setup_data(self, path, batch_size=512, shuffle=False, device='cpu'):
        dataset= data.TCRDataset(path, self.aa_tokenizer, device=device, mhctok=self.mhc_tokenizer)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.all2allmhc_collate_function)
        self.recorder.add_dataset(os.path.basename(path),
                                 {'device': str(device),
                                  'shuffle': shuffle,
                                  'batch_size': batch_size,
                                  'path': path})
        self.dataset = dataset
        self.dataloader = dataloader
    def setup_runtime(self, pad_to=50, special_chars=[2,3,4]):
        self.runtime_pad_to = pad_to
        self.runtime_special_chars = special_chars
    
    @staticmethod
    def build_rollout_mask(mask, input, special_chars=[2,3,4]):
        rollout_mask = mask.clone()
        for special_char in special_chars:
            rollout_mask[input == special_char] = 0
        return rollout_mask
    @staticmethod
    def LLLoss_raw(predictions, targets, ignore_index):
        criterion = nn.NLLLoss(ignore_index=ignore_index, reduction='none')
        if len(targets)>0:
            predictions = predictions[:, :-1, :].contiguous()
            targets = targets[:, 1:]
            bs = targets.shape[0]

            rearranged_output = predictions.view(predictions.shape[0]*predictions.shape[1], -1)
            rearranged_target = targets.contiguous().view(-1)

            loss = criterion(rearranged_output, rearranged_target).reshape(bs,-1).sum(dim=1)
        else:
            loss = torch.zeros(1)
        return loss
    
    def run_one_epoch(self, peptide, alpha, beta, binder, mhc, pad_to=50, special_chars=[2, 3, 4]):
        peptide_input, peptide_mask = peptide['input_ids'], peptide["attention_mask"]
        alpha_input, alpha_mask = alpha['input_ids'], alpha["attention_mask"]
        beta_input, beta_mask = beta['input_ids'], beta["attention_mask"]
        labels = binder
        _, _, prediction_scoresE = self.tulip_track(groundtruths=(alpha_input,beta_input,peptide_input),
                                                    input_ids=(alpha_input,beta_input,peptide_input),
                                                    attention_mask=(alpha_mask,beta_mask,peptide_mask),
                                                    labels=labels,
                                                    mhc=mhc,
                                                    output_hidden_states=True,
                                                    output_attentions=True)
        predictionsE = F.log_softmax(prediction_scoresE, dim=2)

        losse = self.LLLoss_raw(predictionsE, peptide_input, self.aa_tokenizer.pad_token_id)
        score = torch.stack([losse[i] for i in range(len(losse))])
        preds = -1*score

        rollouts = [rollout.cpu().detach() for rollout in self.tulip_track.rollouts(discard_ratio=self.discard_ratio)]
        e_rollout_mask = self.build_rollout_mask(peptide_mask, peptide_input, special_chars=special_chars).cpu()
        a_rollout_mask = self.build_rollout_mask(alpha_mask, alpha_input, special_chars=special_chars).cpu()
        b_rollout_mask = self.build_rollout_mask(beta_mask, beta_input, special_chars=special_chars).cpu()
        rollouts[0] *= a_rollout_mask
        rollouts[1] *= b_rollout_mask
        rollouts[2] *= e_rollout_mask
        rollouts[3] *= a_rollout_mask
        rollouts[4] *= b_rollout_mask
        rollouts[5] *= e_rollout_mask
        rollouts = [np.pad(rollout.numpy(), pad_width=(0, pad_to-len(rollout[0]))) for rollout in rollouts]
        
        preds = preds.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        peptide_seq = [i.replace(' ', '') if len(i) > 0 else None for i in self.aa_tokenizer.batch_decode(peptide_input, skip_special_tokens=True)]
        alpha_seq = [i.replace(' ', '') if len(i) > 0 else None for i in self.aa_tokenizer.batch_decode(alpha_input, skip_special_tokens=True)]
        beta_seq = [i.replace(' ', '') if len(i) > 0 else None for i in self.aa_tokenizer.batch_decode(beta_input, skip_special_tokens=True)]
        
        columns = ['chain_alpha', 'chain_beta', 'chain_epitope', 'binder', 'score',
                'embed_alpha', 'embed_beta', 'embed_epitope',
                'decode_alpha', 'decode_beta', 'decode_epitope',
                'encode_alpha', 'encode_beta', 'encode_epitope']
        alpha_input = alpha_input.cpu().detach().numpy()
        beta_input = beta_input.cpu().detach().numpy()
        peptide_input = peptide_input.cpu().detach().numpy()
        _df = pd.DataFrame([
            alpha_seq, beta_seq, peptide_seq, labels, preds,
            np.pad(alpha_input, pad_width=(0, pad_to-len(alpha_input[0]))),
            np.pad(beta_input, pad_width=(0, pad_to-len(beta_input[0]))),
            np.pad(peptide_input, pad_width=(0, pad_to-len(peptide_input[0]))),
            *rollouts
        ]).T
        _df.columns = columns
        return _df
    
    def run(self, path='model_output'):
        pad_to=self.runtime_pad_to
        special_chars=self.runtime_special_chars
        path = self.path(path)
        os.makedirs(path, exist_ok=True)
        curr_id = 0
        curr_file = os.path.join(path, f'{curr_id}.json')
        rollout_dfs = pd.DataFrame()
        total_added = 0
        bar = tqdm(self.dataloader)
        for peptide, alpha, beta, binder, mhc in bar:
            df = self.run_one_epoch(peptide, alpha, beta, binder, mhc, pad_to=pad_to, special_chars=special_chars)
            total_added += len(df)
            rollout_dfs = pd.concat([rollout_dfs, df])
            if len(rollout_dfs) > 100000:
                rollout_dfs.reset_index(drop=True, inplace=True)
                rollout_dfs.to_json(curr_file)
                curr_id += 1
                curr_file = os.path.join(path, f'{curr_id}.json')
                rollout_dfs = pd.DataFrame()
            bar.set_description(f'{total_added} Rollout Records, {curr_id} Files')
        rollout_dfs.reset_index(drop=True, inplace=True)
        rollout_dfs.to_json(curr_file)
        self.recorder.add_exp({
            'total_samples': total_added,
            'file_number': curr_id,
            'pad_to': pad_to,
            'special_chars': special_chars
        })
        
    def __call__(self):
        self.run()
        self.recorder.document.save(self.path('report.html'), format='html')
        with open(self.path('config.json'), 'w') as f: json.dump(self.config_dict, f, indent=4)
    