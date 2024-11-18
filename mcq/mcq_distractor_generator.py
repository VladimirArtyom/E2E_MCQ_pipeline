from .components.paraphrase_question import ParaphraseQuestion
from .components.distractor_generator import DistractorGenerator
from .components.pipelines import GenerationPipeline
from .components.distractor_filters import Distractors_Filter

from typing import List

import re
class GenerateDistractorParaphrase():
    def __init__(this,
            paraphrasePipeline: GenerationPipeline,
            distractorPipeline: GenerationPipeline,
            distractor_filters: Distractors_Filter,
            ):
            this.paraphraseGenerator: ParaphraseQuestion = paraphrasePipeline
            this.distractorGenerator: DistractorGenerator = distractorPipeline
            this.filters: Distractors_Filter = distractor_filters

    def __call__(this, context: str, question: str, answer: str, n: int=3, **kwargs):
        distractors_1 = this._clean_distractor_1(this.generate_distractor(context=context, question=question, answer=answer, **kwargs))
        outputs, all_outputs = this.filters(answer, distractors_1)
        if len(outputs) >= n:
            return outputs[:n], all_outputs
        return outputs, all_outputs


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
    

class GenerateDistractorsCombineWithAllNoParaphrase():
    def __init__(
        this,
        distractorPipeline: GenerationPipeline,
        distractorAllPipeline: GenerationPipeline,
        distractor_filters: Distractors_Filter
    ):
        this.distractorGenerator: DistractorGenerator = distractorPipeline
        this.distractorAllGenerator: DistractorGenerator = distractorAllPipeline
        this.filters = distractor_filters

    def __call__(this, context: str, question: str, answer: str, n: int=5, **kwargs):
        distractors: List = this._clean_distractors_all(this._generate_distractors_all(context=context, answer=answer, question=question, **kwargs))
        distractors.extend(
            this._clean_distractors_1(this._generate_distractor_1(context=context, answer=answer, question=question, **kwargs))
        )
        outputs, all_outputs = this.filters(answer, distractors)
        if len(outputs) >= n:
            return outputs[:n], all_outputs
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
                cleaned.append(re.sub(pattern, "", distractor))
        return cleaned

    def _clean_distractors_1(this, distractors: List[str]) -> List[str]:
        cleaned = []
        pattern = "<[^>]+>"
        for distractor in distractors:
                cleaned.append(re.sub(pattern, "", distractor))
        return cleaned

class GenerateDistractorsCombineWithAll():
    def __init__(this,
                 paraphrasePipeline: GenerationPipeline,
                 distractorPipeline: GenerationPipeline,
                 distractor_filters: Distractors_Filter,
                 distractorAllPipeline: GenerationPipeline = None,
                 ):
            this.paraphraseGenerator: ParaphraseQuestion = paraphrasePipeline
            this.distractorGenerator: DistractorGenerator = distractorPipeline
            this.distractorAllGenerator: DistractorGenerator = distractorAllPipeline
            this.filters: Distractors_Filter = distractor_filters

    def __call__(this, context: str, question: str, answer: str, fast:bool=True, n: int=3, **kwargs):
        distractors_1 = this._clean_distractor_1(this.generate_distractor(context=context, question=question, answer=answer, **kwargs))
        if not fast:
            distractor_all_kwargs = kwargs.get("kwargs_distractor_all")
            distractors_all = this._clean_distractor_all(this.generate_distractors(context=context, question=question, answer=answer, **distractor_all_kwargs))
            distractors_1.extend(distractors_all)
        outputs, all_outputs = this.filters(answer, distractors_1)
        if len(outputs) >= n:
            return outputs[:n], all_outputs
        return outputs, all_outputs


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
