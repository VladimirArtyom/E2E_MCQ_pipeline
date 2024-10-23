import numpy as np 
from typing import List, Mapping, Any
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from numpy.linalg import norm
import pickle
from typing import List
from numpy import dot

class CosineSimilarity():
    def __init__(this, model_path: str):
        this.model = this._load_model(model_path) 
    
    def __call__(this, target_sentence: str, list_sentences: List[str]):
        pass

    def _load_model(this, model_path: str):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    def _sentence_embedding(this, sentence: str):
        words = sentence.split(" ")
        word_vectors = []

        for word in words:
            if word in this.model:
                word_vectors.append(this.model[word])

        if word_vectors:
            return np.mean(word_vectors, axis=0)
                
        return np.zeros(this.model.vector_size)

    def _calculate_similarity(this, sentence_1: str, sentence_2: str):
        embedding_value_1: float = this._sentence_embedding(sentence_1)
        embedding_value_2: float = this._sentence_embedding(sentence_2)

        return cosine_similarity([embedding_value_1], [embedding_value_2])[0][0]

    def _calculate_similarities(this, sentence_1: str, sentences: List[str]):
        result: Mapping[str, Mapping[str, Any]] = {}
        result["correct"] = {
            "sentence": sentence_1,
            "status": "correct"
        }
        for indx, distractor_candidate in enumerate(sentences):
            distractor = {
                "sentence": distractor_candidate,
                "status": "distractor",
                "similarity": this._calculate_similarity(sentence_1, distractor_candidate)
            }
            result["distractor_{}".format(indx)] = distractor
        
        return result

    def _packed_result(this, result: List[str]):
        ...
    

if __name__ == "__main__":
    embedding_path: str = "./embeddings/1M_embeddings.pkl"
    csc = CosineSimilarity(embedding_path)
    sentence_1 = "kecerdasan digital"
    sentence_2 = ["kecerdasan buatan", "digitalisasi global", "analisis kemajuan", "yaelah"]
    similarity = csc._calculate_similarities(sentence_1, sentence_2 )

    #model = KeyedVectors.load_word2vec_format(embedding_path, binary=False, limit=1000000)
    #with open("./embeddings/1M_embeddings.pkl", "wb") as f:
    #    pickle.dump(model, f)
    

    