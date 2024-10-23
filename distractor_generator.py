from transformers import T5ForConditionalGeneration, T5Tokenizer
from pipelines import GenerationPipeline
from helpers import E2E_TOKEN
from typing import Mapping, Tuple
from torch import Tensor
class DistractorGenerator(GenerationPipeline):
    def __init__(this,
                 model: T5ForConditionalGeneration,
                 tokenizer: T5Tokenizer,
                 max_length: int,
                 question_token: str,
                 context_token: str,
                 answer_token: str,
                 sep_token: str,
                 device: str,
                 ):
        this.model = model
        this.tokenizer = tokenizer
        this.device = device
        this.max_length = max_length

        this.question_token = question_token
        this.context_token = context_token
        this.answer_token = answer_token
        this.sep_token = sep_token
    
    def __call__(this, question: str, context: str, answer: str, **kwargs ):
        input_ids, attention_mask = this._prep_distractor_inputs(question=question,
                                                   context=context,
                                                   answer=answer)
        generated_result = this._generate(input_ids=input_ids,
                                          attention_mask=attention_mask, **kwargs)
        decoded = this._decode(generated_result)

        return decoded
    def _prep_distractor_inputs(this,
                                question: str,
                                context: str,
                                answer: str) -> Tuple[Tensor, Tensor]:

        input_text = "{} {} {} {} {} {}".format(
            this.answer_token,
            answer,
            this.question_token,
            question,
            this.context_token,
            context
        )
        input_ids, attention_mask = this._encode_inputs(input_text)

        return input_ids, attention_mask
        
    def _encode_inputs(this, text) -> Tuple[Tensor, Tensor]:
        encoded = this.tokenizer(
            text,
            max_length=this.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encoded["input_ids"], encoded["attention_mask"]
