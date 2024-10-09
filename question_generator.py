from helpers import E2E_TOKEN
from typing import List, Tuple
from rake_nltk import Rake
from transformers import pipeline
from torch import Tensor
import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from pipelines import GenerationPipeline
class QuestionGenerator(GenerationPipeline):
    
    def __init__(this,
                 qg_model,
                 tokenizer,
                 answer_token: str,
                 context_token: str,
                 max_length: int,
                 max_sentence_to_split: int,
                 stopwords: List[str],
                 device: str):
        super().__init__(

            model=qg_model, tokenizer=tokenizer, device=device
        )

        this.answer_token = answer_token
        this.context_token = context_token

        this.max_length = max_length
        #this.qg_model.eval()

        ## Deprecated
        this.regex_pattern = r'\b(?:tahun|ketinggian|populasi|panjang|luas)\s+(\d+[\.,]?\d*)\s*(?:km|meter|m|tahun|jiwa|persegi)?'
        this.spacy_nlp = spacy.load("xx_sent_ud_sm")
        this.rake = Rake(stopwords=stopwords, language="indonesian", min_length=1, max_length=5)

    def __call__(this, context: str, answer: str, **kwargs_qg):
        prep_input = this._prepare_qag_inputs(context, answer)
        input_ids, attention_mask = this._encode_qg_input(prep_input)
        qg_output = this._generate(input_ids=input_ids, attention_mask=attention_mask)
        generated_question = this._decode(qg_output)
        return generated_question

    def _encode_qg_input(this, qg_input_text: str) -> Tuple[Tensor, Tensor]:
        encoded =  this.tokenizer(
            qg_input_text,
            padding="max_length",
            max_length=this.max_length,
            truncation=True,
            return_tensors="pt"
        )
        return encoded["input_ids"].to(this.device), encoded["attention_mask"].to(this.device)

    def _prepare_qag_inputs(this, context: str, answer: str ) -> Tuple[List[str], List[str]]:

        inp_text: str = f"{this.answer_token} {answer} {this.context_token} {context}"
        return inp_text
"""
if __name__ == "__main__":
    with open("stopwords-id.txt", 'r') as f:
        stopwords = f.read()
    
    qg_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    qg_tokenizer = AutoTokenizer.from_pretrained("t5-small")

    qg_tokenizer.add_special_tokens({
        "additional_special_tokens": [E2E_TOKEN.ANSWER_TOKEN.value,
                                      E2E_TOKEN.CONTEXT_TOKEN.value,]
    })

    qg = QuestionGenerator(qg_model=qg_model,
                           tokenizer=qg_tokenizer,
                           answer_token=E2E_TOKEN.QUESTION_TOKEN.value,
                           context_token=E2E_TOKEN.CONTEXT_TOKEN.value, 
                           max_length=512,
                           max_sentence_to_split=0,
                           stopwords=stopwords,
                           device="cpu"
                          )
    text = "In the heart of a bustling city, a small park serves as a serene escape for residents and visitors alike. Towering trees provide shade on warm summer days, while vibrant flowers bloom in an array of colors, creating a picturesque landscape. Families gather for picnics on the grassy lawns, children laugh as they chase one another around the playground, and joggers find solace on the winding paths that meander through the greenery. The gentle sound of birds chirping and leaves rustling in the breeze adds to the tranquility, making it a perfect spot for reflection and relaxation. As the sun sets, the park transforms, with soft lights illuminating the pathways, inviting couples to stroll hand in hand under the twinkling stars. It is a cherished haven in the midst of urban chaos, reminding everyone of the beauty of nature and the importance of taking a moment to unwind."
    answer ="bustling city"
    qg(text, answer)
    
    bert_model = AutoModelForTokenClassification.from_pretrained('cahya/bert-base-indonesian-NER')
    bert_model_tokenizer = AutoTokenizer.from_pretrained('cahya/bert-base-indonesian-NER')

    evaluator_model = AutoModelForSequenceClassification.from_pretrained("VosLannack/QAG_ID_Evaluator")
    tokenizer = AutoTokenizer.from_pretrained("VosLannack/QAG_ID_Evaluator")

    ner_pipeline = pipeline("ner", model=bert_model, tokenizer=bert_model_tokenizer)

    #text= "Di tengah hutan yang lebat, terdapat sebuah desa kecil yang indah bernama Desa Harapan. Masyarakat di desa ini terkenal dengan keramahtamahan dan kebersamaan mereka. Setiap pagi, para penduduk desa berkumpul di alun-alun untuk menikmati sarapan bersama sambil berdiskusi tentang berbagai hal, mulai dari hasil panen hingga kegiatan sehari-hari. Selain itu, mereka juga memiliki tradisi unik, yaitu merayakan Festival Budaya setiap tahun yang diisi dengan pertunjukan seni dan lomba masak. Hal ini membuat desa tersebut selalu ramai dikunjungi oleh wisatawan dari berbagai daerah."
    #result = ner_pipeline(text)
    #multi_word_entities = aggregate_entities(result)
    #print(multi_word_entities)

    #qg.generate_qag_pairs(text)

"""