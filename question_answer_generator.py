from helpers import E2E_TOKEN
from pipelines import GenerationPipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch import Tensor
from typing import Tuple 
class QuestionAnswerGenerator(GenerationPipeline):
    def __init__(this,
                 model: T5ForConditionalGeneration,
                 tokenizer: T5Tokenizer,
                 max_length: int,
                 context_token: str,
                 device: str,
                 **kwargs
                 ):
        super().__init__(model, tokenizer, device, **kwargs)
        this.max_length = max_length
        this.context_token = context_token
        this.device = device

    def __call__(this, context: str, **kwargs):
        prepared_input = this.prep_input(context)
        input_ids, attention_mask = this._encode(prepared_input)
        output =this._generate(input_ids=input_ids,
                       attention_mask=attention_mask, 
                       **kwargs
                       )
        return this._decode(output )
    
    def prep_input(this, context: str) -> str:
        inp_text = "{} {}".format(this.context_token, context)
        return inp_text

    def _encode(this, context: str) -> Tuple[Tensor, Tensor]:
        encoded = this.tokenizer(
            context,
            max_length=this.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return encoded["input_ids"].to(this.device), encoded["attention_mask"].to(this.device)
    
"""
if __name__ == "__main__":
    qg_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    qg_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    qg_tokenizer.add_special_tokens({
        "additional_special_tokens": [
                                      E2E_TOKEN.QUESTION_TOKEN.value,
                                      E2E_TOKEN.ANSWER_TOKEN.value,
                                      E2E_TOKEN.CONTEXT_TOKEN.value,
                                      E2E_TOKEN.SEP_TOKEN.value
                                      ]
    })

    dg = QuestionAnswerGenerator(
        qg_model,
        qg_tokenizer,
        max_length=512,
        context_token=E2E_TOKEN.CONTEXT_TOKEN,
        device="cpu"
    )
    
    answer = "Tripping"
    question = "What is what is Bro ?"
    context = "In the heart of a bustling city, a small park serves as a serene escape for residents and visitors alike. Towering trees provide shade on warm summer days, while vibrant flowers bloom in an array of colors, creating a picturesque landscape. Families gather for picnics on the grassy lawns, children laugh as they chase one another around the playground, and joggers find solace on the winding paths that meander through the greenery. The gentle sound of birds chirping and leaves rustling in the breeze adds to the tranquility, making it a perfect spot for reflection and relaxation. As the sun sets, the park transforms, with soft lights illuminating the pathways, inviting couples to stroll hand in hand under the twinkling stars. It is a cherished haven in the midst of urban chaos, reminding everyone of the beauty of nature and the importance of taking a moment to unwind."

    print(dg(context=context))
"""