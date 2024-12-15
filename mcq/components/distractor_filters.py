from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Set
from scipy.spatial.distance import jensenshannon
from string import punctuation
import numpy as np
import Levenshtein
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Set
from scipy.spatial.distance import jensenshannon
from string import punctuation
import numpy as np
import Levenshtein
import pickle
import json

class Distractors_Filter():
    def __init__(this, embedding_path):
        this.embedding_model = this._load_embedding(embedding_path)

    def __call__(this, correct_answer: str, distractors: List[str]):
        tout_ds = distractors
        ds = this._length_filter(correct_answer, distractors)
        ds = this._distractors_candidate_filter(correct_answer, ds)
        #ds = this._same_distractor_filter(ds)
        tout_ds = this._distractors_candidate_filter(correct_answer, tout_ds)
        return ds, tout_ds

    def _load_embedding(this, model_path: str):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def _sentence_embedding(this, sentence: str) -> np.ndarray:
        words = sentence.split(" ")
        word_vectors = []

        for word in words: 
            if word in this.embedding_model:
                word_vectors.append(this.embedding_model[word])
        
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        
        return np.zeros(this.embedding_model.vector_size)
    
    def _calculate_cosine_similarity(this, correct_answer: str, distractor: str) -> float:
        embedding_correct = this._sentence_embedding(correct_answer)
        embedding_distractor = this._sentence_embedding(distractor)

        return cosine_similarity([embedding_correct], [embedding_distractor])[0][0]

    def _generate_ngrams(this, text: str, n: int)-> set :
        words = text.split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n + 1)]
        return set(ngrams)


    def _calculate_levenshtein_similarity(this, correct_answer: str, distractor: str):
        distance = Levenshtein.distance(correct_answer, distractor) 
        similarity = distance / (max(len(correct_answer), len(distractor)))
        return similarity
    

    def _distractor_similarity(this, correct_answer: str,
                            distractor: str,
                            threshold: float =0.80,
                            ) -> bool:

        cos_sim = ((this._calculate_cosine_similarity(correct_answer, distractor.strip()) + 1 ) / 2 )

        return cos_sim >= threshold, cos_sim

    def _length_filter(this, correct_answer, distractors: List[Tuple[str, str]], max_length_diff=3):
        correct_len = len(correct_answer.split())
        filtered_distractor =  [d for d in distractors if abs(len(d[0].split()) - correct_len) <= max_length_diff and d[0] != ""]
        return filtered_distractor

    def _distractors_candidate_filter(this, correct_answer, distractors: List[Tuple[str, str]]):
        result = []
        for d in distractors:
            cond, score = this._distractor_similarity(d[0], correct_answer)
            if cond:
                result.append((d, score))
        return sorted(result, reverse=True, key=lambda x : x[1])

    def _same_distractor_filter(this, distractors: List[Tuple[Tuple[str, str], float]], threshold: float = 0.6):
        result = []
        for d1 in distractors:
            for d2 in distractors:
                cosine_sim = this._calculate_cosine_similarity(d1[0][0], d2[0][0])
                if cosine_sim >= threshold:
                    if d1[1] > d2[1] and d1 not in result:
                        result.append(d1)
                    elif d1[1] < d2[1] and d2 not in result:
                        result.append(d2)
        return result
    

    #def _clean_text(this, text: str) -> str :
    #    text = text.lower()
    #    text = text.strip()
    #    text = text.translate(str.maketrans("", "", punctuation))
    #    return text