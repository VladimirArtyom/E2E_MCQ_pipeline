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
                                          attention_mask=attention_mask)
        decoded = this._decode(generated_result, **kwargs)
        print(decoded)
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

    dg = DistractorGenerator(
        qg_model,
        qg_tokenizer,
        max_length=512,
        question_token=E2E_TOKEN.QUESTION_TOKEN,
        context_token=E2E_TOKEN.CONTEXT_TOKEN,
        answer_token=E2E_TOKEN.ANSWER_TOKEN,
        sep_token=E2E_TOKEN.SEP_TOKEN,
        device="cpu"
    )
    
    answer = "Tripping"
    question = "What is what is Bro ?"
    context = "In the heart of a bustling city, a small park serves as a serene escape for residents and visitors alike. Towering trees provide shade on warm summer days, while vibrant flowers bloom in an array of colors, creating a picturesque landscape. Families gather for picnics on the grassy lawns, children laugh as they chase one another around the playground, and joggers find solace on the winding paths that meander through the greenery. The gentle sound of birds chirping and leaves rustling in the breeze adds to the tranquility, making it a perfect spot for reflection and relaxation. As the sun sets, the park transforms, with soft lights illuminating the pathways, inviting couples to stroll hand in hand under the twinkling stars. It is a cherished haven in the midst of urban chaos, reminding everyone of the beauty of nature and the importance of taking a moment to unwind."

    dg(answer=answer, question=question, context=context)
    
"""