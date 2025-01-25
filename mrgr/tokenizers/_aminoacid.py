from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
import os

def amino_acids(weight=None):
    if weight is None:
        _path = os.path.dirname(__file__)
        weight = os.path.join(_path, 'weights/aatok')
    tokenizer = AutoTokenizer.from_pretrained(weight)
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    if tokenizer.sep_token is None: tokenizer.add_special_tokens({'sep_token': '<MIS>'})
    if tokenizer.cls_token is None: tokenizer.add_special_tokens({'cls_token': '<CLS>'})
    if tokenizer.eos_token is None: tokenizer.add_special_tokens({'eos_token': '<EOS>'})
    if tokenizer.mask_token is None: tokenizer.add_special_tokens({'mask_token': '<MASK>'})
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<CLS> $A <EOS>",
        pair="<CLS> $A <MIS> $B:1 <EOS>:1",
        special_tokens=[
            ("<EOS>", 2), ("<CLS>", 3), ("<MIS>", 4),
        ],
    )
    return tokenizer