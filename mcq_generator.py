from argparse import Namespace, ArgumentParser
from question_generator import QuestionGenerator
from question_answer_evaluator import QuestionAnswerEvaluator
from question_answer_generator import QuestionAnswerGenerator
from paraphrase_question import ParaphraseQuestion
from BERT_NER_extractor import NER_extractor
from pipelines import GenerationPipeline

from transformers import T5ForConditionalGeneration, AutoTokenizer, BertForSequenceClassification, BertTokenizer, BertForTokenClassification

import re


def parse_argument() -> Namespace:
    
    args = ArgumentParser()
    args.add_argument("--answer_token",type=str, default="<answer>")
    args.add_argument("--context_token", type=str, default="<context>")
    args.add_argument("--question_token", type=str, default="<question>")
    args.add_argument("--paraphraser_token", type=str, default="<paraphrase>")
    args.add_argument("--device", type=str, default="cpu")

    args.add_argument("--qg_max_split_sentence", type=int, default=64)
    args.add_argument("--qae_use_evaluator")
    args.add_argument("--qag_path", type=str, default="VosLannack/QAG_ID_Generator_t5_base")
    args.add_argument("--qae_path", type=str, default="VosLannack/QAG_ID_Evaluator")
    args.add_argument("--qg_path", type=str, default="VosLannack/QG_ID_Generator_t5_base")
    args.add_argument("--dg_path", type=str, default="VosLannack/Distractor_all_t5-base")
    args.add_argument("--dg_1_path", type=str, default="VosLannack/Distractor_1_t5-small")
    args.add_argument("--ner_path", type=str, default="cahya/bert-base-indonesian-NER")
    args.add_argument("--paraphrase_path", type=str, default="Wikidepia/IndoT5-base-paraphrase")

    return args.parse_args()

class GenerateDistractors():
    def __init__(this,
                 paraphrasePipeline: GenerationPipeline,
                 distractorPipeline: GenerationPipeline,
                 distractorAllPipeline: GenerationPipeline,
                 ):
            this.paraphrasePipeline: ParaphraseQuestion = paraphrasePipeline
            this.distractorPipeline = distractorPipeline
            this.distractorAllPipeline = distractorAllPipeline
    def paraphrase_question(this, question) -> str:
        ...
    
    def calculate_similarity(this, correct_answer: str, distractor: str):
        ...
    
    def generate_distractors(this, context: str, answer: str, question: str):
        ...

    def generate_distractor(this, context: str, answer: str, question: str):
        ...

    ## paraphrase include within this module

class GenerateQuestionAnswerPairs():
    ## Evaluator include in here
    def __init__(this,
                 questionGenerator: GenerationPipeline,
                 questionAnswerGenerator: GenerationPipeline,
                 questionAnswerEvaluator: GenerationPipeline,
                ner: NER_extractor
                 ):
        this.questionGenerator: QuestionGenerator = questionGenerator
        this.questionAnswerGenerator: QuestionAnswerGenerator = questionAnswerGenerator
        this.questionAnswerEvaluator: QuestionAnswerEvaluator = questionAnswerEvaluator
        this.ner_extractor = ner

    def generate_question_answer_pairs(this, context):
        qag_outputs = this.generate_question_from_QAG(context)
        qg_outputs = this.generate_question_from_QG(context)
        
    def _generate_candidate_answers_given_a_sentence(this, sentence):
        answers = this.ner_extractor(sentence)
        return answers

    def generate_question_from_QG(this, context: str):
        #sentences = this._split_context_into_sentence(context)
        pairs = []
        answer_candidates = this._generate_candidate_answers_given_a_sentence(context)
        for answer in answer_candidates:
            potential_question = this.generate_question_from_model(context, answer)
            pairs.append((potential_question,answer))
        return pairs

    def generate_question_from_QAG(this, context: str):
        sentences = this._split_context_into_sentence(context)
        pairs = []
        for sentence in sentences:
            answer, question = this.generate_question_answer_from_model(sentence)
            pairs.append((question, answer))
        return pairs
    
    def _split_context_into_sentence(this, context: str):
        pattern = "(?<=[.!?]) +"
        sentences = re.split(pattern, context)
        return sentences

    def _split_answer_question_pairs_from_qg_model(this, result: str, sep: str = "<question>"):
        res = result.split(sep)
        if len(res) != 2:
            raise ValueError("the QAG model should return Question Answer pairs ,returned: ", res)
        answer = this._clean_qag_answer(res[0])
        question = this._clean_qag_question(res[1])
        return answer, question

    def _clean_qag_answer(this, answer_raw: str) -> str:
        pattern = r"^(?:.*<answer>\s*)(.*)$"
        answer = re.findall(pattern, answer_raw)
        if answer is None or answer == []:
            raise ValueError("Please check the extracted answer")
        return answer[0]

    def _clean_qag_question(this, question_raw: str) -> str:
        pattern = r"(.*)(?:</s>)$"
        question = re.findall(pattern, question_raw)
        if question is None or question == []:
            raise ValueError("Something wrong with the extracted question", question)
        return question[0]
        
    def generate_question_from_model(this, context: str, answer: str, **kwargs):
        return this.questionGenerator(context, answer, **kwargs)

    def generate_question_answer_from_model(this, context: str, **kwargs):
        qag = this.questionAnswerGenerator(context, **kwargs)[0]
        answer, question = this._split_answer_question_pairs_from_qg_model(qag)
        return answer, question
        

    def validator_calculate_pairs_probability(this, question_context: str, answer: str, **kwargs):
        return this.questionAnswerEvaluator(question_context, answer)

if __name__ == "__main__":
    # Use your access token here
    args = parse_argument()
    # Move to document reader for .env token
    # 
    token="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"

    qag_model = T5ForConditionalGeneration.from_pretrained(args.qag_path, use_auth_token=token)
    qag_tokenizer = AutoTokenizer.from_pretrained(args.qag_path, use_auth_token=token)

    qg_model = T5ForConditionalGeneration.from_pretrained(args.qg_path, use_auth_token=token)
    qg_tokenizer = AutoTokenizer.from_pretrained(args.qg_path, use_auth_token=token)

    qae_model = BertForSequenceClassification.from_pretrained(args.qae_path, use_auth_token=token)
    qae_tokenizer = AutoTokenizer.from_pretrained(args.qae_path, use_auth_token=token)

    dg_model = T5ForConditionalGeneration.from_pretrained(args.dg_path, use_auth_token=token)
    dg_tokenizer = AutoTokenizer.from_pretrained(args.dg_path, use_auth_token=token)

    dg_1_model = T5ForConditionalGeneration.from_pretrained(args.dg_1_path, use_auth_token=token)
    dg_1_tokenizer = AutoTokenizer.from_pretrained(args.dg_1_path, use_auth_token=token)

    ner_model = BertForTokenClassification.from_pretrained(args.ner_path)
    ner_tokenizer = BertTokenizer.from_pretrained(args.ner_path)
    ner = NER_extractor(ner_model, ner_tokenizer)

    qag_pipeline = QuestionAnswerGenerator(qag_model,
                                         qag_tokenizer,
                                         device=args.device,
                                         max_length=512,
                                         context_token=args.context_token)

    qg_pipeline = QuestionGenerator(qg_model, qg_tokenizer,
                                    answer_token=args.answer_token,
                                    context_token=args.context_token,
                                    max_length=512,
                                    device=args.device)
    qae_pipeline = QuestionAnswerEvaluator(
        qae_model,
        qae_tokenizer,
        512,
        args.device
    )
    
    QG_generator = GenerateQuestionAnswerPairs(
        questionAnswerGenerator=qag_pipeline,
        questionGenerator=qg_pipeline,
        questionAnswerEvaluator=qae_pipeline,
        ner=ner
    )

    context = "Sejarah Indonesia dimulai sejak zaman prasejarah ketika manusia pertama kali mendiami Nusantara sekitar 1,5 juta tahun yang lalu. Salah satu peradaban paling awal yang tercatat adalah Kerajaan Kutai di Kalimantan Timur pada abad ke-4 Masehi. Kemudian muncul kerajaan-kerajaan besar lainnya seperti Sriwijaya di Sumatera dan Majapahit di Jawa, yang masing-masing mencapai puncak kejayaannya pada abad ke-7 dan ke-14. Pengaruh Hindu-Buddha sangat kuat pada periode ini, yang kemudian diikuti oleh masuknya agama Islam pada abad ke-13 melalui para pedagang dari Arab dan Gujarat. Kolonialisasi Eropa dimulai pada abad ke-16 dengan kedatangan Portugis, diikuti oleh Belanda yang menjajah Indonesia selama lebih dari 300 tahun. Indonesia akhirnya meraih kemerdekaan pada 17 Agustus 1945 setelah perjuangan panjang melawan penjajahan."
    pairs = QG_generator.generate_question_from_QG(context)
    print(pairs)
