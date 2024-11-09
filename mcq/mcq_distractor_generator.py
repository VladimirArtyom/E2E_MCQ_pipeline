from .components.paraphrase_question import ParaphraseQuestion
from .components.distractor_generator import DistractorGenerator
from .components.pipelines import GenerationPipeline
from .components.cosine_similarity import CosineSimilarity

from typing import List

import re

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

    def __call__(this, context: str, question: str, answer: str, fast:bool=True,**kwargs):
        distractors_1 = this._clean_distractor_1(this.generate_distractor(context=context, question=question, answer=answer, **kwargs))
        if not fast:
            distractor_all_kwargs = kwargs.get("kwargs_distractor_all")
            distractors_all = this._clean_distractor_all(this.generate_distractors(context=context, question=question, answer=answer, **distractor_all_kwargs))
            distractors_1.extend(distractors_all)
        outputs = this.calculate_similarities(answer, distractors_1)
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
