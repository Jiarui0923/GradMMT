
import torch
from torch import nn
from transformers import PretrainedConfig
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_outputs import ModelOutput
from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from typing import Optional, Tuple, Union, Any, Dict

class ED_LMOutput(ModelOutput):
    clf_loss: Optional[torch.FloatTensor] = None
    clf_logits: Optional[torch.FloatTensor] = None
    decoder_outputsA = None
    encoder_outputsA = None
    decoder_outputsB = None
    encoder_outputsB = None
    decoder_outputsE = None
    encoder_outputsE = None

class Tulip(PreTrainedModel):
    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"


    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoderA: Optional[PreTrainedModel] = None,
        decoderA: Optional[PreTrainedModel] = None,
        encoderB: Optional[PreTrainedModel] = None,
        decoderB: Optional[PreTrainedModel] = None,
        encoderE: Optional[PreTrainedModel] = None,
        decoderE: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoderA is None or decoderA is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoderA.config, decoderA.config)
            #config = EncoderDecoderConfig.from_encoder_decoder_configs(encoderA.config, decoderA.config,encoderB.config, decoderB.config,encoderE.config, decoderE.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if encoderA is None:
            from ..auto.modeling_auto import AutoModel

            encoderA = AutoModel.from_config(config.encoder)
        if encoderE is None:
            from ..auto.modeling_auto import AutoModel

            encoderE = AutoModel.from_config(config.encoder)
        if encoderB is None:
            from ..auto.modeling_auto import AutoModel

            encoderB = AutoModel.from_config(config.encoder)

        if decoderA is None:
            from ..auto.modeling_auto import AutoModelForCausalLM
            decoderA = AutoModelForCausalLM.from_config(config.decoder)
        if decoderB is None:
            from ..auto.modeling_auto import AutoModelForCausalLM
            decoderB = AutoModelForCausalLM.from_config(config.decoder)
        if decoderE is None:
            from ..auto.modeling_auto import AutoModelForCausalLM
            decoderE = AutoModelForCausalLM.from_config(config.decoder)
        self.reweight=False
        self.encoderA = encoderA
        self.decoderA = decoderA
        self.encoderB = encoderB
        self.decoderB = decoderB
        self.encoderE = encoderE
        self.decoderE = decoderE
        self.num_labels = 2
        self.MLMHeadA =  BertOnlyMLMHead(decoderA.config)
        self.MLMHeadB =  BertOnlyMLMHead(decoderB.config)
        self.MLMHeadE =  BertOnlyMLMHead(decoderE.config)


        # Miss Mask Implemetation
        self.skipMiss = True
        self.MissA = nn.Parameter(torch.zeros((1,encoderA.config.hidden_size)), requires_grad=True)
        self.MissB = nn.Parameter(torch.zeros((1,encoderB.config.hidden_size)), requires_grad=True)
        self.MissE = nn.Parameter(torch.zeros((1,encoderE.config.hidden_size)), requires_grad=True)
        # This classifier is only here for potential future supervised task
        self.classifier = nn.Linear(3*decoderA.config.hidden_size, 2)
        self.mhc_embeddings = nn.Embedding(encoderA.config.mhc_vocab_size, encoderA.config.hidden_size)

        if (
            self.encoderA.config.hidden_size != self.decoderA.config.hidden_size
            and self.decoderA.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.encoderA.config.hidden_size, self.decoderA.config.hidden_size)

        if self.encoderA.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoderA} should not have a LM Head. Please use a model without LM Head"
            )

        self.tie_weights()

    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def get_encoder(self, encoder_name='B'):
        if encoder_name=='A':
            return self.encoderA
        elif encoder_name=='B':
            return self.encoderB
        elif encoder_name=='E':
            return self.encoderE


    def get_decoder(self, decoder_name='B'):
        if decoder_name=='A':
            return self.decoderA
        elif decoder_name=='B':
            return self.decoderB
        elif decoder_name=='E':
            return self.decoderE



    def get_input_embeddings(self, encoder_name='B'):
        if encoder_name=='A':
            return self.encoderA.get_input_embeddings()
        elif encoder_name=='B':
            return self.encoderB.get_input_embeddings()
        elif encoder_name=='E':
            return self.encoderE.get_input_embeddings()
        


    def get_output_embeddings(self, decoder_name='B'):
        if decoder_name=='A':
            return self.decoderA.get_output_embeddings()
        elif decoder_name=='B':
            return self.decoderB.get_output_embeddings()
        elif decoder_name=='E':
            return self.decoderE.get_output_embeddings()
        


    def set_output_embeddings(self, new_embeddings, decoder_name='B'):
        if decoder_name=='A':
            return self.decoderA.set_output_embeddings(new_embeddings)
        elif decoder_name=='B':
            return self.decoderB.set_output_embeddings(new_embeddings)
        elif decoder_name=='E':
            return self.decoderE.set_output_embeddings(new_embeddings)
        


    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        decoder = kwargs_decoder.pop("model", None)
        
        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        return cls(encoder=encoder, decoder=decoder, config=config)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = (None, None,None),
        attention_mask: Optional[torch.FloatTensor] =  (None, None,None),
        # decoder_input_ids: Optional[torch.LongTensor] =  (None, None,None),
        # decoder_attention_mask: Optional[torch.BoolTensor] =  (None, None,None),
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = (None, None,None),
        past_key_values: Tuple[Tuple[torch.FloatTensor]] =  (None, None,None),
        inputs_embeds: Optional[torch.FloatTensor] = (None, None,None),
        # decoder_inputs_embeds: Optional[torch.FloatTensor] =  (None, None,None),
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] =  (None, None,None),
        output_attentions: Optional[bool] =  None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = (None, None,None),
        mhc=None,
        togenerate=None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        # print('forward', input_ids)
        input_idsA=input_ids[0]
        input_idsB=input_ids[1]
        input_idsE=input_ids[2]


        # Miss mask Implementation
        # To Do replace hard coded 3 and 4 with bos and miss token ids
        if self.skipMiss:
            if input_idsA!= None:
                if input_idsA.shape[1] == 1:
                    MissMaskA = input_idsA.clone().detach()[:,0] != 3
                else:
                    MissMaskA = input_idsA.clone().detach()[:,1] == 4
            if input_idsB!= None:
                if input_idsB.shape[1] == 1:
                    MissMaskB = input_idsB.clone().detach()[:,0] != 3
                else:
                    MissMaskB = input_idsB.clone().detach()[:,1] == 4

            if input_idsE!= None:
                if input_idsE.shape[1] == 1:
                    MissMaskE = input_idsE.clone().detach()[:,0] != 3
                else:
                    MissMaskE = input_idsE.clone().detach()[:,1] == 4
        
        attention_maskA=attention_mask[0]
        attention_maskB=attention_mask[1]
        attention_maskE=attention_mask[2]

        encoder_outputsA=encoder_outputs[0]
        encoder_outputsB=encoder_outputs[1]
        encoder_outputsE=encoder_outputs[2]

        past_key_valuesA=past_key_values[0]
        past_key_valuesB=past_key_values[1]
        past_key_valuesE=past_key_values[2]

        inputs_embedsA=inputs_embeds[0]
        inputs_embedsB=inputs_embeds[1]
        inputs_embedsE=inputs_embeds[2]
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputsA is None:
            encoder_outputsA = self.encoderA(
                input_ids=input_idsA,
                attention_mask=attention_maskA,
                inputs_embeds=inputs_embedsA,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputsA, tuple):
            encoder_outputsA = BaseModelOutput(*encoder_outputsA)


        if encoder_outputsB is None:
            encoder_outputsB = self.encoderB(
                input_ids=input_idsB,
                attention_mask=attention_maskB,
                inputs_embeds=inputs_embedsB,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputsB, tuple):
            encoder_outputsB = BaseModelOutput(*encoder_outputsB)

                
        if encoder_outputsE is None:
            encoder_outputsE = self.encoderE(
                input_ids=input_idsE,
                attention_mask=attention_maskE,
                inputs_embeds=inputs_embedsE,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputsE, tuple):
            encoder_outputsE = BaseModelOutput(*encoder_outputsE)

        encoder_hidden_statesA = encoder_outputsA[0]
        encoder_hidden_statesB = encoder_outputsB[0]
        encoder_hidden_statesE = encoder_outputsE[0]
        # optionally project encoder_hidden_states


        # Miss mask Implementation
        if self.skipMiss:
            if input_idsA != None:
                encoder_hidden_statesA = encoder_hidden_statesA.clone()
                encoder_hidden_statesA[MissMaskA,0,:] = self.MissA
                encoder_hidden_statesA[MissMaskA,1,:] = self.MissA
                encoder_hidden_statesA[MissMaskA,2,:] = self.MissA
            if input_idsB != None:
                encoder_hidden_statesB = encoder_hidden_statesB.clone()
                encoder_hidden_statesB[MissMaskB,0,:] = self.MissB
                encoder_hidden_statesB[MissMaskB,1,:] = self.MissB
                encoder_hidden_statesB[MissMaskB,2,:] = self.MissB
            if input_idsE != None:
                encoder_hidden_statesE = encoder_hidden_statesE.clone()
                encoder_hidden_statesE[MissMaskE,0,:] = self.MissE
                encoder_hidden_statesE[MissMaskE,1,:] = self.MissE
                encoder_hidden_statesE[MissMaskE,2,:] = self.MissE



        if (
            self.encoderA.config.hidden_size != self.decoderA.config.hidden_size
            and self.decoderA.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_statesA = self.enc_to_dec_proj(encoder_hidden_statesA)

        if (
            self.encoderB.config.hidden_size != self.decoderB.config.hidden_size
            and self.decoderB.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_statesB = self.enc_to_dec_proj(encoder_hidden_statesB)

        if (
            self.encoderE.config.hidden_size != self.decoderE.config.hidden_size
            and self.decoderE.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_statesE = self.enc_to_dec_proj(encoder_hidden_statesE)

        mhc_encoded = self.mhc_embeddings(mhc["input_ids"])
        mhc_attention_mask = mhc["attention_mask"]
        # Decode
        if togenerate not in ['B','E']:
            labelsA = (labels, input_idsA)
            decoder_outputsA = self.decoderA(
                input_ids = input_idsA,
                attention_mask = attention_maskA,
                encoder_hidden_states = torch.cat([mhc_encoded,encoder_hidden_statesB, encoder_hidden_statesE], dim=1),
                encoder_attention_mask = torch.cat([mhc_attention_mask, attention_maskB, attention_maskE], dim=1),
                inputs_embeds = inputs_embedsA,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                labels=labelsA,
                use_cache=use_cache,
                past_key_values=past_key_valuesA,
                return_dict=return_dict,
                **kwargs_decoder,
            )
            pooled_outputA = decoder_outputsA.pooled_output

        if togenerate not in ['A','E']:
            labelsB = (labels, input_idsB)
            decoder_outputsB = self.decoderB(
                input_ids = input_idsB,
                attention_mask = attention_maskB,
                encoder_hidden_states = torch.cat([mhc_encoded,encoder_hidden_statesA, encoder_hidden_statesE], dim=1),
                encoder_attention_mask = torch.cat([mhc_attention_mask,attention_maskA, attention_maskE], dim=1),
                inputs_embeds = inputs_embedsB,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                labels=labelsB,
                use_cache=use_cache,
                past_key_values=past_key_valuesB,
                return_dict=return_dict,
                **kwargs_decoder,
            )
            pooled_outputB = decoder_outputsB.pooled_output

        if togenerate not in ['A','B']:
            labelsE = (labels, input_idsE)
            decoder_outputsE = self.decoderE(
                input_ids = input_idsE,
                attention_mask = attention_maskE,
                encoder_hidden_states = torch.cat([mhc_encoded,encoder_hidden_statesA, encoder_hidden_statesB], dim=1),
                encoder_attention_mask = torch.cat([mhc_attention_mask,attention_maskA, attention_maskB], dim=1),
                inputs_embeds = inputs_embedsE,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                labels=labelsE,
                use_cache=use_cache,
                past_key_values=past_key_valuesE,
                return_dict=return_dict,
                **kwargs_decoder,
            )
            pooled_outputE= decoder_outputsE.pooled_output

        lossCLS = None
        logits = None
      


        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None

        if togenerate == 'A':
            return Seq2SeqLMOutput(
                loss=loss,
                logits=decoder_outputsA.lm_logits,
                past_key_values=decoder_outputsA.past_key_values,
                decoder_hidden_states=decoder_outputsA.hidden_states,
                decoder_attentions=decoder_outputsA.attentions,
                cross_attentions=decoder_outputsA.cross_attentions,
                encoder_last_hidden_state=torch.cat([mhc_encoded,encoder_hidden_statesB, encoder_hidden_statesE], dim=1),
                encoder_hidden_states=encoder_outputsE.hidden_states,
                encoder_attentions=torch.cat([mhc_attention_mask, attention_maskB, attention_maskE], dim=1),
            )
        elif togenerate == 'B':
            return Seq2SeqLMOutput(
                loss=loss,
                logits=decoder_outputsB.lm_logits,
                past_key_values=decoder_outputsB.past_key_values,
                decoder_hidden_states=decoder_outputsB.hidden_states,
                decoder_attentions=decoder_outputsB.attentions,
                cross_attentions=decoder_outputsB.cross_attentions,
                encoder_last_hidden_state=torch.cat([mhc_encoded,encoder_hidden_statesA, encoder_hidden_statesE], dim=1),
                encoder_hidden_states=encoder_outputsE.hidden_states,
                encoder_attentions=torch.cat([mhc_attention_mask,attention_maskA, attention_maskE], dim=1),
            )
        elif togenerate == 'E':
            return Seq2SeqLMOutput(
                loss=loss,
                logits=decoder_outputsE.lm_logits,
                past_key_values=decoder_outputsE.past_key_values,
                decoder_hidden_states=decoder_outputsE.hidden_states,
                decoder_attentions=decoder_outputsE.attentions,
                cross_attentions=decoder_outputsE.cross_attentions,
                encoder_last_hidden_state=torch.cat([mhc_encoded,encoder_hidden_statesA, encoder_hidden_statesB], dim=1),
                encoder_hidden_states=encoder_outputsE.hidden_states,
                encoder_attentions=torch.cat([mhc_attention_mask,attention_maskA, attention_maskB], dim=1),
            )
        

        else:
            return ED_LMOutput(
                loss = lossCLS,
                clf_logits=logits,
                encoder_outputsA = encoder_outputsA,
                decoder_outputsA = decoder_outputsA,
                encoder_outputsB = encoder_outputsB,
                decoder_outputsB = decoder_outputsB,
                encoder_outputsE = encoder_outputsE,
                decoder_outputsE = decoder_outputsE,
            )

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        if decoder_start_token_id is None:
            raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # print('prepare_decoder_input_ids_from_labels')
        return self.shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=(None, None, None), use_cache=None, encoder_outputs=(None, None, None), **kwargs
    ):

        # print('prepare_inputs_for_generation')
        togenerate = kwargs['togenerate']
        if togenerate == 'A':
            decoder_inputs = self.decoderA.prepare_inputs_for_generation(input_ids, past=past)
            decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
            input_dict = {
                "input_ids": (decoder_inputs["input_ids"], None, None),
                "attention_mask": (decoder_attention_mask,attention_mask[1],attention_mask[2]),
                "encoder_outputs": encoder_outputs,
                "past_key_values": (decoder_inputs["past_key_values"], None, None),
                "use_cache": use_cache,
                "togenerate": togenerate,
                "mhc": kwargs['mhc'],
            }
            return input_dict
        elif togenerate == 'B':
            decoder_inputs = self.decoderB.prepare_inputs_for_generation(input_ids, past=past)
            decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
            input_dict = {
                "input_ids": (None,decoder_inputs["input_ids"], None),
                "attention_mask": (attention_mask[0],decoder_attention_mask,attention_mask[2]),
                "encoder_outputs": encoder_outputs,
                "past_key_values": (None, decoder_inputs["past_key_values"], None),
                "use_cache": use_cache,
                "togenerate": togenerate,
                "mhc": kwargs['mhc'],
            }
            return input_dict
        elif togenerate == 'E':
            decoder_inputs = self.decoderE.prepare_inputs_for_generation(input_ids, past=past)
            decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
            input_dict = {
                "input_ids": (None,None,decoder_inputs["input_ids"]),
                "attention_mask": (attention_mask[0],attention_mask[1],decoder_attention_mask),
                "encoder_outputs": encoder_outputs,
                "past_key_values": (None, None,  decoder_inputs["past_key_values"]),
                "use_cache": use_cache,
                "togenerate": togenerate,
                "mhc": kwargs['mhc'],
            }
            return input_dict
        else:
            raise ValueError('togenerate should be A, B or E')

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)


    def set_reweight(self):
        self.reweight = True


 
    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """

        input_name = "input_ids"
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 5. if `inputs` is still None, try to create `input_ids` from BOS token
        if inputs is None:
            bs = model_kwargs["input_ids"][0].shape[0]
            inputs = torch.ones((bs,1), dtype=torch.long) * bos_token_id

        return inputs, input_name, model_kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:

        encoder_kwargs = model_kwargs.copy()
        encoder_kwargs["togenerate"] = None

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        # encoder_kwargs[model_input_name] = inputs_tensor
        # model_kwargs["encoder_outputs"]: ModelOutput 
        out = self.forward(**encoder_kwargs)
        model_kwargs["encoder_outputs"] = (out.encoder_outputsA, out.encoder_outputsB, out.encoder_outputsE)
        model_kwargs["decoder_input_ids"] = inputs_tensor  #### Not NEEDED?
        model_kwargs.pop("input_ids", None) #### WHY?

        return model_kwargs


    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # print('_update_model_kwargs_for_generation')
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs
    

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        decoder_input_ids_start = torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
        elif self.config.model_type == "vision-encoder-decoder" and "donut" in self.name_or_path.lower():
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (decoder_input_ids[:, 0] != decoder_start_token_id).all().item():
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs


    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = (None,None,None),
        encoder_outputs: Optional[Tuple[ModelOutput]] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        # print('_expand_inputs_for_generation')
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)
        model_kwargs["mhc"]["input_ids"] = model_kwargs["mhc"]["input_ids"].index_select(0, expanded_return_idx)
        model_kwargs["mhc"]["attention_mask"] = model_kwargs["mhc"]["attention_mask"].index_select(0, expanded_return_idx)
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = (attention_mask[0].index_select(0, expanded_return_idx),
                                                attention_mask[1].index_select(0, expanded_return_idx),
                                                attention_mask[2].index_select(0, expanded_return_idx))
        

        if is_encoder_decoder:
            if encoder_outputs == (None,None,None):
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs[0]["last_hidden_state"] = encoder_outputs[0].last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs[0].last_hidden_state.device)
            )
            encoder_outputs[1]["last_hidden_state"] = encoder_outputs[1].last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs[1].last_hidden_state.device)
            )
            encoder_outputs[2]["last_hidden_state"] = encoder_outputs[2].last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs[2].last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

