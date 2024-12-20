from .components.pipelines import GenerationPipeline
from .components.BERT_NER_extractor import NER_extractor
from .components.question_generator import QuestionGenerator
from .components.question_answer_evaluator import QuestionAnswerEvaluator
from .components.question_answer_generator import QuestionAnswerGenerator
from .components.enums import ExperimentQG, Metadata

from typing import List, Tuple
import re

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

    def __call__(this, context, experiment_type: str, **kwargs) -> List[Tuple[str, str]]:
        kwargs_qag = kwargs.get("kwargs_qag")
        kwargs_qg = kwargs.get("kwargs_qg")

        if experiment_type == ExperimentQG.QG_ONLY.value:
            outputs = this._attach_metadata(this.generate_question_from_QG(context, **kwargs_qg),
                                            Metadata.QG)
        elif experiment_type == ExperimentQG.QAG_ONLY.value:
            outputs = this._attach_metadata(this.generate_question_from_QAG(context, **kwargs_qag),
                                            Metadata.QAG)
        else:
            qg_outputs = this._attach_metadata(this.generate_question_from_QG(context, **kwargs_qg), Metadata.QG)
            qag_outputs = this._attach_metadata(this.generate_question_from_QAG(context, **kwargs_qag), Metadata.QAG)
            outputs =  qg_outputs + qag_outputs

        return outputs
        
    def _generate_candidate_answers_given_a_sentence(this, sentence):
        answers = this.ner_extractor(sentence)
        return answers

    def _attach_metadata(this, outputs: List[str], metadata: Metadata):
        result = []
        for output in outputs:
            result.append((output, metadata.name))
        return result


    def generate_question_from_QG(this, context: str, **kwargs):
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
            try:
                answer, question = this.generate_question_answer_from_model(sentence + "<context>" + context, **kwargs)
                pairs.append((question, answer))
            except:
                continue
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