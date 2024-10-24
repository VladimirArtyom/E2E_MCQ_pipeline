from argparse import Namespace, ArgumentParser
from question_generator import QuestionGenerator
from question_answer_evaluator import QuestionAnswerEvaluator
from question_answer_generator import QuestionAnswerGenerator
from distractor_generator import DistractorGenerator
from paraphrase_question import ParaphraseQuestion
from BERT_NER_extractor import NER_extractor
from pipelines import GenerationPipeline
from kwargs import DG_1_KWARGS, DG_ALL_KWARGS, PARAPHRASE_KWARGS, DG_1_KWARGS_N, QG_KWARGS, QAG_KWARGS
from cosine_similarity import CosineSimilarity
from omegaconf import OmegaConf
from typing import List, Tuple, Mapping

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



class Executor():
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
        this.qag_pipeline = QuestionAnswerGenerator(qag_model,
                                                    qag_tokenizer,
                                                    device=args.device,
                                                    max_length=512,
                                                    context_token=args.context_token)
    
    def prepare_qg_pipeline(this):
        qg_model = T5ForConditionalGeneration.from_pretrained(this.qg_path, use_auth_token=token)
        qg_tokenizer = AutoTokenizer.from_pretrained(this.qg_path, use_auth_token=token)
        this.qg_pipeline = QuestionGenerator(
            qg_model,
            qg_tokenizer,
            device=this.args.device,
            answer_token=this.args.answer_token,
            context_token=this.args.context_token,
            max_length=512
        )
    
    def prepare_distractor_pipeline(this):
        dg_model = T5ForConditionalGeneration.from_pretrained(this.dg_path, use_auth_token=token)
        dg_tokenizer = AutoTokenizer.from_pretrained(this.dg_path, use_auth_token=token)
        this.dg_pipeline = DistractorGenerator(
            dg_model,
            dg_tokenizer,
            512,
            question_token=this.args.question_token,
            context_token=this.args.context_token,
            answer_token=this.args.answer_token,
            sep_token=this.args.sep_token,
            device=this.args.device
        )
    
    def prepare_qae_pipeline(this):
        qae_model = BertForSequenceClassification()
        qae_model = BertForSequenceClassification.from_pretrained(args.qae_path, use_auth_token=token)
        qae_tokenizer = AutoTokenizer.from_pretrained(args.qae_path, use_auth_token=token)
        this.qae_pipeline = QuestionAnswerEvaluator(
            qae_model,
            qae_tokenizer,
            512,
            args.device
        )

    def prepare_paraphrase_pipeline(this):
        paraphrase_model = T5ForConditionalGeneration.from_pretrained(this.paraphrase_path)
        paraphrase_tokenizer = AutoTokenizer.from_pretrained(this.paraphrase_path)
        this.paraphrase_pipeline = ParaphraseQuestion(
            paraphrase_model,
            paraphrase_tokenizer,
            512,
            args.device
        )      

    def prepare_distractor_1_pipeline(this):
        dg_1_model = T5ForConditionalGeneration.from_pretrained(this.dg_1_path, use_auth_token=token)
        dg_1_tokenizer = AutoTokenizer.from_pretrained(this.dg_1_path, use_auth_token=token)
        this.dg_1_pipeline = DistractorGenerator(
            dg_1_model,
            dg_1_tokenizer,
            128,
            question_token=this.args.question_token,
            context_token=this.args.context_token,
            answer_token=this.args.answer_token,
            sep_token=this.args.sep_token,
            device=this.args.device
        )

    def prepare_ner_pipeline(this):
        ner_model = BertForTokenClassification.from_pretrained(this.ner_path)
        ner_tokenizer = BertTokenizer.from_pretrained(this.ner_path)
        this.ner = NER_extractor(ner_model, ner_tokenizer)

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
            this.prepare_qae_pipeline()

        if this.paraphrase_path:
            this.prepare_paraphrase_pipeline()

        this.prepare_qag_pipeline()
        this.prepare_qg_pipeline()
        this.prepare_distractor_pipeline()
        this.prepare_distractor_1_pipeline()
        this.prepare_ner_pipeline()


class GenerateDistractors():
    def __init__(this,
                 paraphrasePipeline: GenerationPipeline,
                 distractorPipeline: GenerationPipeline,
                 cosine_similarity: CosineSimilarity,
                 distractorAllPipeline: GenerationPipeline = None,
                 ):
            this.paraphraseGenerator: ParaphraseQuestion = paraphrasePipeline
            this.distractorGenerator: DistractorGenerator = distractorPipeline
            this.distractorAllGenerator: DistractorGenerator = distractorAllPipeline
            this.cosine = cosine_similarity

    def __call__(this, context: str, question: str, answer: str, **kwargs):
        distractor_all_kwargs = kwargs.get("kwargs_distractor_all")
        distractors_all = this._clean_distractor_all(this.generate_distractors(context=context, question=question, answer=answer, **distractor_all_kwargs))
        #return distractors_all
        #distractors_1 = this._clean_distractor_1(this.generate_distractor(context=context, question=question, answer=answer, **kwargs))
        #return distractors_1
        #distractors_1.extend(distractors_all)
        outputs = this.calculate_similarities(answer, distractors_all)
        return outputs


    def _clean_distractor_all(this, texts: List[str]) -> List[str]: 
        distractors: List[str] = []
        pattern = "<[^>]+>"
        for text in texts:
            split = text.split("<sep>")
            for s in split:
                distractors.append(re.sub(pattern, "" ,s))
        
        return distractors

    def _clean_distractor_1(this, texts: List[str]):
        distractors: List[str] = []
        pattern = "<[^>]+>"
        for text in texts:
           distractors.append(re.sub(pattern, "", text))
        return distractors

    def paraphrase_question(this, question, **kwargs) -> str:
        return this.paraphraseGenerator(question, **kwargs)
    
    def calculate_similarities(this, correct_answer: str, distractors: str):
        return this.cosine(correct_answer, distractors)
    
    def generate_distractors(this, context: str, answer: str, question: str, **kwargs):
        return this.distractorAllGenerator(question, context=context, answer=answer, **kwargs)

    def generate_distractor(this, context: str, answer: str, question: str, **kwargs):
        distractors = []
        kwargs_paraphrase = kwargs.get("kwargs_paraphrase")
        kwargs_distractor = kwargs.get("kwargs_distractor_1")
        if this.paraphraseGenerator is not None:
            paraphrase_ques = this.paraphrase_question(question, **kwargs_paraphrase)
            for para_ques in paraphrase_ques:
                distractors.extend(this.distractorGenerator(question=para_ques, context=context, answer=answer, **kwargs_distractor))
        else:

            distractors.extend(this.distractorGenerator(question=question, context=context, answer=answer, **kwargs_distractor))
        return distractors

    ## paraphrase include within this module

class GenerateQuestionAnswerPairs():
    ## Evaluator include in here
    def __init__(this,
                 questionGenerator: GenerationPipeline,
                 questionAnswerGenerator: GenerationPipeline,
                 ner: NER_extractor,
                 questionAnswerEvaluator: GenerationPipeline = None,
                 ):
        this.questionGenerator: QuestionGenerator = questionGenerator
        this.questionAnswerGenerator: QuestionAnswerGenerator = questionAnswerGenerator
        this.questionAnswerEvaluator: QuestionAnswerEvaluator = questionAnswerEvaluator
        this.ner_extractor = ner

    def __call__(this, context, **kwargs) -> List[Tuple[str, str]]:
        kwargs_qag = kwargs.get("kwargs_qag")
        kwargs_qg = kwargs.get("kwargs_qg")

        qag_outputs = this.generate_question_from_QAG(context, **kwargs_qag)
        qg_outputs = this.generate_question_from_QG(context, **kwargs_qg)
        outputs =  qg_outputs + qag_outputs
        return outputs
        
    def _generate_candidate_answers_given_a_sentence(this, sentence):
        answers = this.ner_extractor(sentence)
        return answers

    def generate_question_from_QG(this, context: str, **kwargs):
        sentences = this._split_context_into_sentence(context)
        pairs = []
        answer_candidates = this._generate_candidate_answers_given_a_sentence(context)
        for answer in answer_candidates:
            potential_question = this.generate_question_from_model(context, answer, **kwargs)[0]
            pairs.append((this._clean_qg_question(potential_question),answer))
        return pairs
    def _clean_qg_question(this, text: str):
        pattern = r"<[^>]+>"
        clean = re.sub(pattern, "", text) 
        return clean

    def generate_question_from_QAG(this, context: str, **kwargs):
        sentences = this._split_context_into_sentence(context)
        pairs = []
        for sentence in sentences:
            answer, question = this.generate_question_answer_from_model(sentence, **kwargs)
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

class MCQ_Generator():
    def __init__(this,
                 qg_generator: GenerateQuestionAnswerPairs,
                 dg_generator: GenerateDistractors,
                 ):
        this.qg_generator = qg_generator
        this.dg_generator = dg_generator
    
    def __call__(this, context: str, **kwargs):
        final_outputs = {}
        list_de_qag = this.qg_generator(context, **kwargs)
        for indx, content in enumerate(list_de_qag):
            question = content[0]
            answer = content[1]
            distractors = this.dg_generator(context, question, answer, **kwargs)
            final_outputs[indx] = {
                "context": context,
                "question": question,
                "answer": answer,
                "distractors": distractors
            }
        return final_outputs

if __name__ == "__main__":
    # Use your access token here
    args = parse_argument()
    # Move to document reader for .env token
    # 
    config = OmegaConf.load("config.yaml")
    token = config.h_token

    qag_model = T5ForConditionalGeneration.from_pretrained(args.qag_path_base, use_auth_token=token)
    qag_tokenizer = AutoTokenizer.from_pretrained(args.qag_path_base, use_auth_token=token)

    qg_model = T5ForConditionalGeneration.from_pretrained(args.qg_path_base, use_auth_token=token)
    qg_tokenizer = AutoTokenizer.from_pretrained(args.qg_path_base, use_auth_token=token)

    #qae_model = BertForSequenceClassification.from_pretrained(args.qae_path, use_auth_token=token)
    #qae_tokenizer = AutoTokenizer.from_pretrained(args.qae_path, use_auth_token=token)

    dg_model = T5ForConditionalGeneration.from_pretrained(args.dg_path_base, use_auth_token=token)
    dg_tokenizer = AutoTokenizer.from_pretrained(args.dg_path_base, use_auth_token=token)

    dg_1_model = T5ForConditionalGeneration.from_pretrained(args.dg_1_path_small, use_auth_token=token)
    dg_1_tokenizer = AutoTokenizer.from_pretrained(args.dg_1_path_small, use_auth_token=token)

    ner_model = BertForTokenClassification.from_pretrained(args.ner_path_base)
    ner_tokenizer = BertTokenizer.from_pretrained(args.ner_path_base)
    ner = NER_extractor(ner_model, ner_tokenizer)

    #paraphrase_model = T5ForConditionalGeneration.from_pretrained(args.paraphrase_path)
    #paraphrase_tokenizer = AutoTokenizer.from_pretrained(args.paraphrase_path)

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


    QG_generator = GenerateQuestionAnswerPairs(
        questionAnswerGenerator=qag_pipeline,
        questionGenerator=qg_pipeline,
        questionAnswerEvaluator=None,
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
    """
    paraphrase_pipeline = ParaphraseQuestion(
        paraphrase_model,
        paraphrase_tokenizer,
        512,
        args.device
    )
    
    """

    DG_generator = GenerateDistractors(
        distractorAllPipeline=dg_all_pipeline,
        distractorPipeline=dg_1_pipeline,
        paraphrasePipeline=None,
        cosine_similarity=CosineSimilarity("./embeddings/1M_embeddings.pkl")
    )
    #context = "Sejarah Indonesia dimulai sejak zaman prasejarah ketika manusia pertama kali mendiami Nusantara sekitar 1,5 juta tahun yang lalu. Salah satu peradaban paling awal yang tercatat adalah Kerajaan Kutai di Kalimantan Timur pada abad ke-4 Masehi. Kemudian muncul kerajaan-kerajaan besar lainnya seperti Sriwijaya di Sumatera dan Majapahit di Jawa, yang masing-masing mencapai puncak kejayaannya pada abad ke-7 dan ke-14. Pengaruh Hindu-Buddha sangat kuat pada periode ini, yang kemudian diikuti oleh masuknya agama Islam pada abad ke-13 melalui para pedagang dari Arab dan Gujarat. Kolonialisasi Eropa dimulai pada abad ke-16 dengan kedatangan Portugis, diikuti oleh Belanda yang menjajah Indonesia selama lebih dari 300 tahun. Indonesia akhirnya meraih kemerdekaan pada 17 Agustus 1945 setelah perjuangan panjang melawan penjajahan."
    #question = "Dimana letak kerajaan Kutai ?"
    #answer = "Kalimantan Timur"
    mcq = MCQ_Generator(
        QG_generator,
        DG_generator
    )
    context = "Amerika Serikat, atau yang sering disebut AS, adalah sebuah negara di Amerika Utara yang terdiri dari 50 negara bagian. Dikenal dengan julukan 'Negeri Paman Sam,' Amerika Serikat memiliki pengaruh besar dalam bidang ekonomi, politik, dan budaya di seluruh dunia. Ibukota negara ini adalah Washington, D.C., sementara New York City dikenal sebagai pusat keuangan global. Dengan berbagai latar belakang budaya yang beragam, negara ini juga dikenal dengan inovasi teknologi, pendidikan tinggi berkualitas, dan keindahan alam yang menakjubkan, mulai dari Grand Canyon hingga Taman Nasional Yellowstone. Amerika Serikat juga merupakan negara yang mendukung kebebasan dan hak asasi manusia, meskipun masih dihadapkan dengan tantangan dalam hal kesetaraan sosial dan ekonomi"
    question = "Siapa yang disebut Paman sam ?"
    answer = "Amerika Serikat"

    kwargs = {
        "kwargs_qg": QG_KWARGS,
        "kwargs_qag":QAG_KWARGS,
        "kwargs_distractor_all": DG_ALL_KWARGS,
        "kwargs_distractor_1": DG_1_KWARGS,
        "kwargs_paraphrase": PARAPHRASE_KWARGS
    }
    ques = mcq(context, **kwargs)
    print(ques)
    #distractors = DG_generator(question, context, answer, **kwargs)
    #print(distractors)
    
    #print("question", question)
    #print("Jawaban benar", answer)
    #rint("Pengecoh",distractors)
    #pairs = QG_generator.generate_question_from_QAG(context)
    #print(pairs)
