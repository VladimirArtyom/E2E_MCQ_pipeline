
import torch
import pandas as pd
import numpy as np
from typing import List, Mapping, Tuple
from rake_nltk import Rake
import re
from transformers import pipeline
import random
from torch import Tensor
import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
class QuestionGenerator:
    
    def __init__(this, qg_model,
                 evaluator_pipeline,
                 bert_ner_pipeline,
                 tokenizer,
                 answer_token: str,
                 context_token: str,
                 max_length: int,
                 max_sentence_to_split: int,
                 stopwords: List[str],
                 device: str):
        this.bert_ner_pipeline = bert_ner_pipeline
        this.evaluator_pipeline = evaluator_pipeline

        this.answer_token = answer_token
        this.context_token = context_token
        this.device = device

        this.qg_tokenizer = tokenizer
        this.qg_model = qg_model

        this.max_length = max_length
        #this.qg_model.eval()

        ## Deprecated
        this.regex_pattern = r'\b(?:tahun|ketinggian|populasi|panjang|luas)\s+(\d+[\.,]?\d*)\s*(?:km|meter|m|tahun|jiwa|persegi)?'
        this.spacy_nlp = spacy.load("xx_sent_ud_sm")
        this.rake = Rake(stopwords=stopwords, language="indonesian", min_length=1, max_length=5)

    def generate_qag_pairs(this, text: str) -> Tuple[List[str], List[str]]:
        inputs: List[str] = []
        answers: List[str] = []
        prep_inputs, prep_answers = this._prepare_qag_inputs(text)
        generated_questions = this._generate_questions(prep_inputs, prep_answers)

    def evaluate_qag_pairs(this):
        ...

    @torch.no_grad()    
    def _generate_question(this, qg_input_text: str) -> List[str]:
        encoded_input = this._encode_qg_input(qg_input_text)
        output = this.qg_model.generate(input_ids=encoded_input["input_ids"])
        question = this.qg_tokenizer.decode(
            output[0],
            skip_special_token=True
        )
        return question

    def _encode_qg_input(this, qg_input_text: str) -> Tensor:
        return this.qg_tokenizer(
            qg_input_text,
            padding="max_length",
            return_attention=False,
            max_length=this.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(this.device)

    def _prepare_qag_inputs(this, text ) -> Tuple[List[str], List[str]]:
        inputs: List[str] = []
        answers: List[str] = []

        candidate_answers = this.get_answer_candidate_entities(text)
        if candidate_answers == "<NO_ANSWER>":
            return -1
        for answer in candidate_answers:
            inp_text: str = f"{this.answer_token} {answer} {this.context_token} {text}"
            answers.append(answer)
            inputs.append(inp_text)
        
        return (inputs, answers)
    def _extract_entity_u_bert_ner():
        ...

    def get_answer_candidate_entities(this, sentence: str):
        candidates_ner = this.bert_ner_pipeline(sentence)
        ## Add code to create it
        if candidates_ner:
            return candidates_ner
        return "<NO-ANSWER>"


    def _deprecated_extract_entity_u_regex(this, sentence: str) -> List[str]:
        entities = []
        matches = re.findall(this.regex_pattern, sentence)
        if matches:
            entities.extend(matches)
        # Needs to be cleaned and convert into a real number
        return entities

    def _deprecated_extract_entity_u_ner(this, sentence: str) -> List[str]:
        entities = []
        docs = list(this.spacy_nlp.pipe([sentence], disable=["parser"]))
        for doc in docs:
            entities.append(doc)

        return entities
    
    def _deprecated_extract_entity_u_rake(this, sentence) -> List[str]:
        entities = []
        this.rake.extract_keywords_from_text(sentence)
        keywords = this.rake.get_ranked_phrases()
        if keywords:
           entities.extend(keywords)
        return entities

    def generate(this,
                 input_text: str,
                 num_questions: int = None,
                 use_evaluator: bool = True):
        
        print("Generating questions ...\n\n")
    


def aggregate_entities(ner_results):
    aggregated_entities = []
    current_entity = None
    
    for token in ner_results:
        entity_type = token['entity'].split('-')[-1]  # Handle B-ORG or I-ORG
        if current_entity and token['entity'].startswith("I-") and entity_type == current_entity['entity']:
            # If the token is part of the same entity, add to the existing one
            current_entity['word'] += " " + token['word']
        else:
            # If it's a new entity or the previous entity ended, save it and start a new one
            if current_entity:
                aggregated_entities.append(current_entity)
            current_entity = {"entity": entity_type, "word": token['word']}
    
    # Append the last entity if it exists
    if current_entity:
        aggregated_entities.append(current_entity)
    
    return aggregated_entities

if __name__ == "__main__":
    with open("stopwords-id.txt", 'r') as f:
        stopwords = f.read()
    
    qg_model = AutoModelForSeq2SeqLM.from_pretrained("VosLannack/QG_ID_Generator")
    qg_tokenizer = AutoTokenizer.from_pretrained("VosLannack/QG_ID_Generator")

    bert_model = AutoModelForTokenClassification.from_pretrained('cahya/bert-base-indonesian-NER')
    bert_model_tokenizer = AutoTokenizer.from_pretrained('cahya/bert-base-indonesian-NER')

    evaluator_model = AutoModelForSequenceClassification.from_pretrained("VosLannack/QAG_ID_Evaluator")
    tokenizer = AutoTokenizer.from_pretrained("VosLannack/QAG_ID_Evaluator")

    ner_pipeline = pipeline("ner", model=bert_model, tokenizer=bert_model_tokenizer)

    qg = QuestionGenerator(None, None, None, None, 512, 64, stopwords, "cuda")
    #text= "Di tengah hutan yang lebat, terdapat sebuah desa kecil yang indah bernama Desa Harapan. Masyarakat di desa ini terkenal dengan keramahtamahan dan kebersamaan mereka. Setiap pagi, para penduduk desa berkumpul di alun-alun untuk menikmati sarapan bersama sambil berdiskusi tentang berbagai hal, mulai dari hasil panen hingga kegiatan sehari-hari. Selain itu, mereka juga memiliki tradisi unik, yaitu merayakan Festival Budaya setiap tahun yang diisi dengan pertunjukan seni dan lomba masak. Hal ini membuat desa tersebut selalu ramai dikunjungi oleh wisatawan dari berbagai daerah."
    #result = ner_pipeline(text)
    #multi_word_entities = aggregate_entities(result)
    #print(multi_word_entities)

    #qg.generate_qag_pairs(text)

