from transformers import AutoTokenizer
import os

def mhc(enable=False, weight=None):
    if weight is None:
        _path = os.path.dirname(__file__)
        if enable: weight = os.path.join(_path, 'weights/mhctok')
        else: weight = os.path.join(_path, 'weights/nomhctok')
    mhctok = AutoTokenizer.from_pretrained(weight)
    return mhctok
    