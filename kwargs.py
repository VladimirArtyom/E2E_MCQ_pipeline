from typing import Mapping, Any
from enum import Enum
QAG_KWARGS: Mapping[str, Any] = {
    "num_beams": 5,
    "top_p": 0.90,
    "top_k": 250,
    "temperature": 1.4,
    "max_length": 512,
    "num_return_sequences": 1,
    "repetition_penalty": 2.5,
    "do_sample": True,
    "early_stopping":True,
}

QG_KWARGS: Mapping[str, Any] = {
    "num_beams": 3,
    "top_p": 0.98,
    "top_k": 100,
    "temperature": 0.8,
    "max_length": 512,
    "num_return_sequences": 1,
    "repetition_penalty": 1.5,
    "no_repeat_ngram_size": 2,
    "early_stopping":True,
    "do_sample": True
}

DG_1_KWARGS: Mapping[str, Any] = {
    "num_beams": 10,
    "top_p": 0.98,
    "top_k": 130,
    "temperature": 1.2,
    "max_length": 512,
    "num_return_sequences": 10,
    "repetition_penalty": 1.5,
    "no_repeat_ngram_size": 2,
    "do_sample": True,
    "early_stopping":True,
}

DG_ALL_KWARGS: Mapping[str, Any] = {

    "num_beams": 5,
    "top_p": 0.95,
    "top_k": 150,
    "repetition_penalty":2.5,
    "temperature": 1.5,
    "max_length": 512,
    "do_sample":True,
    "num_return_sequences": 3,
    "early_stopping":True,
}


PARAPHRASE_KWARGS: Mapping[str, Any] = {
        "num_beams": 3,
        "top_p": 0.98,
        "top_k": 120,
        "num_return_sequences":5,
        "repetition_penalty":1.2,
        "temperature": 1.8,
        "max_length": 256,
        "no_repeat_ngram_size": 3,
        "early_stopping":True,
        "do_sample": True
}

