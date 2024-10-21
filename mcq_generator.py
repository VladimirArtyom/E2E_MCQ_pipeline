from argparse import Namespace, ArgumentParser
from question_generator import QuestionGenerator
from question_answer_evaluator import QuestionAnswerEvaluator
from question_answer_generator import QuestionAnswerGenerator
from distractor_generator import DistractorGenerator
from paraphrase_question import ParaphraseQuestion
from BERT_NER_extractor import NER_extractor
from pipelines import GenerationPipeline
from kwargs import DG_1_KWARGS, DG_ALL_KWARGS

from transformers import T5ForConditionalGeneration, AutoTokenizer, BertForSequenceClassification, BertTokenizer, BertForTokenClassification

import re


def parse_argument() -> Namespace:
    
    args = ArgumentParser()
    args.add_argument("--answer_token",type=str, default="<answer>")
    args.add_argument("--context_token", type=str, default="<context>")
    args.add_argument("--question_token", type=str, default="<question>")
    args.add_argument("--sep_token", type=str, default="<sep>")
    args.add_argument("--paraphraser_token", type=str, default="<paraphrase>")
    args.add_argument("--device", type=str, default="cpu")

    args.add_argument("--qg_max_split_sentence", type=int, default=64)
    args.add_argument("--use_evaluator", type=int, default=0, choices=[0, 1])
    args.add_argument("--use_paraphrase", type=int, default=0, choices=[0, 1])
    args.add_argument("--fast_execution", type=int, default=0, choices=[0, 1])
    args.add_argument("--qag_path_base", type=str, default="VosLannack/QAG_ID_Generator_t5_base")
    args.add_argument("--ner_path_base", type=str, default="cahya/bert-base-indonesian-NER")
    args.add_argument("--qg_path_base", type=str, default="VosLannack/QG_ID_Generator_t5_base")
    args.add_argument("--qg_path_small", type=str, default="VosLannack/QG_ID_Generator_t5_small")
    args.add_argument("--dg_path_base", type=str, default="VosLannack/Distractor_all_t5-base")
    args.add_argument("--dg_path_small", type=str, default="VosLannack/Distractor_all_t5_small")
    args.add_argument("--qae_path", type=str, default="VosLannack/QAG_ID_Evaluator")
    args.add_argument("--dg_1_path_small", type=str, default="VosLannack/Distractor_1_t5-small")
    args.add_argument("--paraphrase_path", type=str, default="Wikidepia/IndoT5-base-paraphrase")

    return args.parse_args()

class MCQ_Generator():
    def __init__(this,
                 args: Namespace,
                 use_evaluator: bool = 0,
                 use_paraphrase: bool = 0,
                 fast_execution: bool = 0,
                 ):
        
        this.use_evaluator = use_evaluator
        this.use_paraphrase = use_paraphrase
        this.fast_execution = fast_execution
        this.args = args

        this.evaluator_path = None
        this.paraphrase_path = None

        this.qg_path = None
        this.qag_path = None
        this.dg_path = None
        this.dg_1_path = None

    def prepare_qag_pipeline(this):
        qag_model = T5ForConditionalGeneration.from_pretrained(this.qag_path, use_auth_token=token)
        qag_tokenizer = AutoTokenizer.from_pretrained(this.qag_path, use_auth_token=token)
        ...
    
    def prepare_qg_pipeline(this):
        qg_model = T5ForConditionalGeneration.from_pretrained(this.qg_path, use_auth_token=token)
        qg_tokenizer = AutoTokenizer.from_pretrained(this.qg_path, use_auth_token=token)
        ...
    
    def prepare_distractor_pipeline(this):
        dg_model = T5ForConditionalGeneration.from_pretrained(this.dg_path, use_auth_token=token)
        dg_tokenizer = AutoTokenizer.from_pretrained(this.dg_path, use_auth_token=token)
        ...
    
    def prepare_qae_pipeline(this):
        ...

    def prepare_paraphrase_pipeline(this):
        paraphrase_model = T5ForConditionalGeneration.from_pretrained(this.paraphrase_path)
        paraphrase_tokenizer = AutoTokenizer.from_pretrained(this.paraphrase_path)
        ...

    def prepare_distractor_1_pipeline(this):
        dg_1_model = T5ForConditionalGeneration.from_pretrained(this.dg_1_path, use_auth_token=token)
        dg_1_tokenizer = AutoTokenizer.from_pretrained(this.dg_1_path, use_auth_token=token)
        ...

    def prepare_ner_pipeline(this):
        ner_model = BertForTokenClassification.from_pretrained(this.ner_path)
        ner_tokenizer = BertTokenizer.from_pretrained(this.ner_path)
        ner = NER_extractor(ner_model, ner_tokenizer)



    def prepare_model(this):
        if this.use_evaluator:
            this.evaluator_path = this.args.qae_path
        
        if this.use_paraphrase:
            this.paraphrase_path = this.args.paraphrase_path
        
        if this.fast_execution:
            this.qg_path = this.args.qg_path_small
            # Currently use the base for QAG, i forgot to trained it on small
            this.qag_path = this.args.qag_path_base
            this.dg_path = this.args.dg_path_small
            this.dg_1_path = this.args.dg_path_1_small
        else:
            this.qg_path = this.args.qg_path_base
            this.qag_path = this.args.qag_path_base
            this.dg_path = this.args.dg_path_base
            ## Currently using the small for single distractors
            this.dg_1_path = this.args.dg_path_1_small

        if this.evaluator_path:
            qae_model = BertForSequenceClassification.from_pretrained(this.evaluator_path, use_auth_token=token)
            qae_tokenizer = AutoTokenizer.from_pretrained(this.evaluator_path, use_auth_token=token)
            this.prepare_qae_pipeline()

        if this.paraphrase_path:
            this.prepare_paraphrase_pipeline()

class GenerateDistractors():
    def __init__(this,
                 paraphrasePipeline: GenerationPipeline,
                 distractorPipeline: GenerationPipeline,
                 distractorAllPipeline: GenerationPipeline,
                 ):
            this.paraphraseGenerator: ParaphraseQuestion = paraphrasePipeline
            this.distractorGenerator: DistractorGenerator = distractorPipeline
            this.distractorAllGenerator: DistractorGenerator = distractorAllPipeline

    def paraphrase_question(this, question) -> str:
        return this.paraphraseGenerator(question)
    
    def calculate_similarity(this, correct_answer: str, distractor: str):
        ...
    
    def generate_distractors(this, context: str, answer: str, question: str):
        return this.distractorAllGenerator(question, context=context, answer=answer)

    def generate_distractor(this, context: str, answer: str, question: str):
        return this.distractorGenerator(question=question, context=context, answer=answer)

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

    #qae_model = BertForSequenceClassification.from_pretrained(args.qae_path, use_auth_token=token)
    #qae_tokenizer = AutoTokenizer.from_pretrained(args.qae_path, use_auth_token=token)

    dg_model = T5ForConditionalGeneration.from_pretrained(args.dg_path, use_auth_token=token)
    dg_tokenizer = AutoTokenizer.from_pretrained(args.dg_path, use_auth_token=token)

    #dg_1_model = T5ForConditionalGeneration.from_pretrained(args.dg_1_path, use_auth_token=token)
    #dg_1_tokenizer = AutoTokenizer.from_pretrained(args.dg_1_path, use_auth_token=token)

    ner_model = BertForTokenClassification.from_pretrained(args.ner_path)
    ner_tokenizer = BertTokenizer.from_pretrained(args.ner_path)
    ner = NER_extractor(ner_model, ner_tokenizer)

    #paraphrase_model = T5ForConditionalGeneration.from_pretrained(args.paraphrase_path)
    #paraphrase_tokenizer = AutoTokenizer.from_pretrained(args.paraphrase_path)
    """
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
    
    dg_1_pipeline = DistractorGenerator(
        dg_1_model,
        dg_1_tokenizer,
        512,
        args.question_token,
        args.context_token,
        args.answer_token,
        args.sep_token,
        args.device
    )
    
    dg_all_pipeline = DistractorGenerator(
        dg_model,
        dg_tokenizer,
        512,
        args.question_token,
        args.context_token,
        args.answer_token,
        args.sep_token,
        args.device
    )

    paraphrase_pipeline = ParaphraseQuestion(
        paraphrase_model,
        paraphrase_tokenizer,
        512,
        args.device
    )
    

    DG_generator = GenerateDistractors(
        distractorAllPipeline=dg_all_pipeline,
        distractorPipeline=dg_1_pipeline,
        paraphrasePipeline=paraphrase_pipeline
    )

    """
    context = "Sejarah Indonesia dimulai sejak zaman prasejarah ketika manusia pertama kali mendiami Nusantara sekitar 1,5 juta tahun yang lalu. Salah satu peradaban paling awal yang tercatat adalah Kerajaan Kutai di Kalimantan Timur pada abad ke-4 Masehi. Kemudian muncul kerajaan-kerajaan besar lainnya seperti Sriwijaya di Sumatera dan Majapahit di Jawa, yang masing-masing mencapai puncak kejayaannya pada abad ke-7 dan ke-14. Pengaruh Hindu-Buddha sangat kuat pada periode ini, yang kemudian diikuti oleh masuknya agama Islam pada abad ke-13 melalui para pedagang dari Arab dan Gujarat. Kolonialisasi Eropa dimulai pada abad ke-16 dengan kedatangan Portugis, diikuti oleh Belanda yang menjajah Indonesia selama lebih dari 300 tahun. Indonesia akhirnya meraih kemerdekaan pada 17 Agustus 1945 setelah perjuangan panjang melawan penjajahan."
    question = "Kolonialisasi Eropa dimulai pada abad ke-16 dengan kedatangan negara apa?"
    answer = "Portugis"
    #paraphrased_question = DG_generator.paraphrase_question(question)
    #print(paraphrased_question)
    #distractors = DG_generator.distractorAllGenerator(question, context, answer, **DG_ALL_KWARGS)
    #print("question", question)
    #print("Jawaban benar", answer)
    #rint("Pengecoh",distractors)
    #pairs = QG_generator.generate_question_from_QAG(context)
    #print(pairs)
