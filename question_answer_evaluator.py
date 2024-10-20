from typing import Mapping 
from torch import Tensor
from transformers import BertForSequenceClassification, AutoTokenizer

class QuestionAnswerEvaluator():

    def __init__(this, 
                 model: BertForSequenceClassification,
                 tokenizer: AutoTokenizer,
                 max_length: int,
                 device: str,
                 ):
        this.model = model
        this.tokenizer = tokenizer
        this.max_length = max_length,
        this.device = device

    def __call__(this, question: str, answer: str, **kwargs):
        input_ids, attention_mask = this._encode(question=question, answer=answer)
        generated_raw = this._generate(input_ids=input_ids, attention_mask=attention_mask)
        print(generated_raw)
        return generated_raw
        
    def _decode(this, outputs: Tensor, **kwargs):
        decode_results = []
        for output in outputs:
            out = this.tokenizer.decode(output,
                                        skip_specialtokens=True,
                                        clean_up_tokenizations_spaces=True,
                                        **kwargs
                                        )
            decode_results.append(out)
            
        return decode_results

    def _generate(this, input_ids: Tensor, attention_mask: Tensor):
        params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        output = this.model(**params)
        return output

    def _encode(this ,question: str, answer: str) -> Mapping[str, Tensor]:
        encoded = this.tokenizer(
            text=question,
            text_pair=answer,
            max_length=this.max_length,
            padding="max_length",
            truncation=True,
        )

        return encoded["input_ids"], encoded["attention_mask"]