from typing import Mapping, Any
QAG_KWARGS: Mapping[str, Any] = {
    "num_beams": 10,
    "top_p": 0.95,
    "top_k": 200,
    "temperature": 2.0,
    "max_length": 512,
    "num_return_sequences": 1,
    "repetition_penalty": 4.5,
    "early_stopping":True,
}

QG_KWARGS: Mapping[str, Any] = {
    "num_beams": 10,
    "top_p": 0.95,
    "top_k": 200,
    "temperature": 1.0,
    "max_length": 64,
    "num_return_sequences": 1,
    "repetition_penalty": 3.5,
    "early_stopping":True,
}

DG_1_KWARGS: Mapping[str, Any] = {
    "num_beams": 10,
    "top_p": 0.98,
    "top_k": 200,
    "temperature": 0.8,
    "max_length": 64,
    "num_return_sequences": 1,
    "repetition_penalty": 4.5,
    "early_stopping":True,
}

DG_ALL_KWARGS: Mapping[str, Any] = {

    "num_beams": 10,
    "top_p": 0.8,
    "top_k": 250,
    "repetition_penalty":3.5,
    "temperature": 2.5,
    "max_length": 512,
    "num_return_sequences": 1,
    "early_stopping":True,
}

PARAPHRASE_KWARGS: Mapping[str, Any] = {
    "num_beams": 15,
    "top_p": 0.90,
    "top_k": 200,
    "num_return_sequences":1,
    "repetition_penalty":4.2,
    "temperature": 1.5,
    "max_length": 128,
    "num_return_sequences": 5,
    "early_stopping":True,
}
