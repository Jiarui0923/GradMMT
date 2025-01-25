from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertConfig

from ._tulip_petal import TulipPetal
from ._bert_last_pooler import BertLastPooler
from ._tulip_core import Tulip as TulipCore



class Tulip(nn.Module):
    def __init__(self, config, max_length=50, aa_vocab_size=30, mhc_vocab_size=30, pad_id=-1):
        super().__init__()
        encoderA, encoderB, encoderE = self._build_encoder(config=config, max_length=max_length,
                                                           aa_vocab_size=aa_vocab_size, mhc_vocab_size=mhc_vocab_size,
                                                           pad_id=pad_id)
        decoderA, decoderB, decoderE = self._build_decoder(config=config, max_length=max_length,
                                                           aa_vocab_size=aa_vocab_size, pad_id=pad_id)
        self.model = TulipCore(encoderA=encoderA, encoderB=encoderB, encoderE=encoderE,
                               decoderA=decoderA, decoderB=decoderB, decoderE=decoderE)
        
    def load_state_dict(self, *args, **kwargs):
        self.model.load_state_dict(*args, **kwargs)
        
    def _build_encoder(self, config, max_length=50, aa_vocab_size=30, mhc_vocab_size=30, pad_id=-1):
        encoder_config = BertConfig(vocab_size = aa_vocab_size,
                        max_position_embeddings = max_length, # this shuold be some large value
                        num_attention_heads = config["num_attn_heads"],
                        num_hidden_layers = config["num_hidden_layers"],
                        hidden_size = config["hidden_size"],
                        type_vocab_size = 1,
                        pad_token_id =  pad_id)
        encoder_config.mhc_vocab_size = mhc_vocab_size
        encoderA = BertModel(config=encoder_config)
        encoderB = BertModel(config=encoder_config)
        encoderE = BertModel(config=encoder_config)
        return encoderA, encoderB, encoderE

    def _build_decoder(self, config, max_length=50, aa_vocab_size=30, pad_id=-1):
        decoder_config = BertConfig(vocab_size = aa_vocab_size,
                        max_position_embeddings = max_length, # this shuold be some large value
                        num_attention_heads = config["num_attn_heads"],
                        num_hidden_layers = config["num_hidden_layers"],
                        hidden_size = config["hidden_size"],
                        type_vocab_size = 1,
                        is_decoder=True,
                        pad_token_id =  pad_id)    # Very Important
        decoder_config.add_cross_attention=True
        decoderA = TulipPetal(config=decoder_config) #BertForMaskedLM
        decoderA.pooler = BertLastPooler(config=decoder_config)
        decoderB = TulipPetal(config=decoder_config) #BertForMaskedLM
        decoderB.pooler = BertLastPooler(config=decoder_config)
        decoderE = TulipPetal(config=decoder_config) #BertForMaskedLM
        decoderE.pooler = BertLastPooler(config=decoder_config)
        return decoderA, decoderB, decoderE
        
    def forward(self, **kwargs):
        out = self.model(**kwargs)
        prediction_scoresE = out.decoder_outputsE.lm_logits
        predictionsE = F.log_softmax(prediction_scoresE, dim=2)
        prediction_scoresA = out.decoder_outputsA.lm_logits
        predictionsA = F.log_softmax(prediction_scoresA, dim=2)
        prediction_scoresB = out.decoder_outputsB.lm_logits
        predictionsB = F.log_softmax(prediction_scoresB, dim=2)
        return predictionsA, predictionsB, predictionsE
