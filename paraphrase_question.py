from helpers import E2E_TOKEN
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pipelines import GenerationPipeline
from torch import Tensor
from typing import Tuple
class ParaphraseQuestion(GenerationPipeline):
    
    def __init__(this, 
                 model: T5ForConditionalGeneration,
                 tokenizer: T5Tokenizer,
                 max_length: int,
                 device: str,
            ):
        super().__init__(model=model, tokenizer=tokenizer, device=device)
        this.max_length = max_length
    
    def __call__(this, question: str, **kwargs):
        input_ids, attention_mask = this._encode(this._prep_input(question)) 
        paraphrased_question_raw = this._generate(input_ids, attention_mask)
        return this._decode(paraphrased_question_raw)

    def _prep_input(this, text: str):
        return "paraphrase: {} </s>".format(text)

    def _encode(this, sentence: str) -> Tuple[Tensor, Tensor]:
        encoded: Tensor = this.tokenizer(
            sentence,
            max_length=this.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encoded["input_ids"], encoded["attention_mask"]
    
"""
if __name__ == "__main__":
    qg_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    qg_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    pq = ParaphraseQuestion(model=qg_model,
                            tokenizer=qg_tokenizer,
                            max_length=128,
                            device="cpu")
    
    question = "Anak anak melakukan piket kelas agar kebersihan kelas terjaga"
    text = pq(question=question)
    print(text)
    
"""