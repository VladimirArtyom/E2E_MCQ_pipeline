from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch import Tensor

from typing import List

class GenerationPipeline:
    def __init__(this,
                 model: T5ForConditionalGeneration,
                 tokenizer: T5Tokenizer,
                 device: str,
                 **kwargs):
        this.model = model
        this.tokenizer = tokenizer
        this.device = device



        this.additional_kwargs = kwargs

    def __call__(this, **kwargs):
        raise NotImplementedError
    
    def _generate(this, input_ids: Tensor, attention_mask: Tensor, **kwargs) -> Tensor:
        params = {"input_ids": input_ids,
                  "attention_mask": attention_mask,
                  **kwargs
                  }

        outputs = this.model.generate(**params)

        return outputs

    def _decode(this, outputs: Tensor, **kwargs) -> List[str]:
        decode_results: List = []
        
        for output in outputs:
            sentence = this.tokenizer.decode(
                output,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
                **kwargs
            )
            decode_results.append(sentence)

        return decode_results
            
"""

class QuestionAnswerGenerationPipeline(T5GenerationPipeline):
    def __call__(this):
        ...

class DistractorGenerationPipeline(T5GenerationPipeline):
    def __call__(this, **kwargs):
        ...

class QuestionGenerationPipeline(T5GenerationPipeline):
    def __call__(this, **kwargs):
        ...

class QuestionParaphraserPipeline(T5GenerationPipeline):
    def __call__(this, **kwargs):
        ...

"""