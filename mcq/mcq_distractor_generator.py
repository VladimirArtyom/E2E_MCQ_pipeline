from .components.paraphrase_question import ParaphraseQuestion
from .components.distractor_generator import DistractorGenerator
from .components.pipelines import GenerationPipeline
from .components.distractor_graders import Distractors_Grader
from .components.enums import *
from typing import List

import re

class GenerateDistractorsCombineWithAll():
    def __init__(
        this,
        distractorPipeline: GenerationPipeline,
        distractorAllPipeline: GenerationPipeline,
        distractor_grader: Distractors_Grader,
        paraphrasePipeline: ParaphraseQuestion
    ):
        this.distractorGenerator: DistractorGenerator = distractorPipeline
        this.distractorAllGenerator: DistractorGenerator = distractorAllPipeline
        this.paraphrasePipeline: ParaphraseQuestion = paraphrasePipeline
        this.graders = distractor_grader

    def __call__(this, context: str, question: str, answer: str, experiment_type: str, n: int=10, **kwargs):
        if experiment_type == ExperimentDG.DAG_ONLY.value:
            distractors: List = this._clean_distractors_all(this._generate_distractors_all(context=context, answer=answer,
                                                                                           question=question, **kwargs))
            distractors = this._attach_metadata(distractors, Metadata.DAG.value)
        elif experiment_type == ExperimentDG.DG_ONLY.value:
            kwargs_paraphrase = kwargs.get("kwargs_paraphrase")
            paraphrased_questions = this.paraphrasePipeline(question, **kwargs_paraphrase)
            distractors: List = []
            for pp_question in paraphrased_questions:
                if pp_question != "<UNK>":
                    distractors_current = []
                    distractors_current.extend(this._clean_distractors_1(this._generate_distractor_1(context=context, answer=answer,
                                                                                        question=pp_question, **kwargs)))
                    distractors.extend(this._attach_metadata(distractors_current, pp_question))
            distractors = this._attach_metadata(distractors, Metadata.DG.value)
            distractors.extend(
                this._attach_metadata(this._clean_distractors_1(this._generate_distractor_1(context=context, answer=answer, question=question, **kwargs)), Metadata.DG.value)
            )
        else:
            kwargs_paraphrase = kwargs.get("kwargs_paraphrase")
            paraphrased_questions = this.paraphrasePipeline(question, **kwargs_paraphrase)
            distractors: List = []
            for pp_question in paraphrased_questions:
                if pp_question != "<UNK>":
                    distractors_current = []
                    distractors_current.extend(this._clean_distractors_1(this._generate_distractor_1(context=context, answer=answer,
                                                                                        question=pp_question, **kwargs)))
                    distractors.extend(this._attach_metadata(distractors_current, pp_question))
            distractors = this._attach_metadata(distractors, Metadata.DG.value)
            distractors.extend(
                this._attach_metadata(this._clean_distractors_1(this._generate_distractor_1(context=context, answer=answer, question=question, **kwargs)), Metadata.DG.value)
            )
            distractors.extend(this._attach_metadata(this._clean_distractors_all(this._generate_distractors_all(context=context, answer=answer, question=question, **kwargs)), Metadata.DAG.value))

        outputs, all_outputs = this.graders(answer, distractors)
        
        return outputs, all_outputs


    def _generate_distractors_all(this, context: str, answer: str, question: str, **kwargs) -> List:
        distractor_all_kwargs = kwargs.get("kwargs_distractor_all")
        return this.distractorAllGenerator(question, context=context, answer=answer, **distractor_all_kwargs)

    def _generate_distractor_1(this, context: str, answer: str, question: str, **kwargs) -> List:
        kwargs_distractor = kwargs.get("kwargs_distractor_1")
        return this.distractorGenerator(question=question, context=context, answer=answer, **kwargs_distractor)

    def _clean_distractors_all(this, distractors: List[str]) -> List[str]:
        cleaned = []
        pattern = "<[^>]+>"
        for text in distractors:
            list_of_distractor = text.split("<sep>") # Put the <sep> token here

            for distractor in list_of_distractor:
                ds  = distractor.split("</s>")
                if ds != []:
                    for d in ds:
                        if d != " " and d != "":
                            cleaned.append(re.sub(pattern,"",(d).strip()))
                else:
                    if distractor != " " and distractor != "":
                        cleaned.append(re.sub(pattern,"",(distractor).strip()))
        return cleaned

    def _clean_distractors_1(this, distractors: List[str]) -> List[str]:
        cleaned = []
        pattern = "<[^>]+>"
        for distractor in distractors:
                cleaned.append(re.sub(pattern, "", distractor))
        return cleaned

    def _attach_metadata(this, distractors: List[str], metadata_name: str) -> List[str]:
        result = []
        for distractor in distractors:
            result.append((distractor, metadata_name))
        return result
