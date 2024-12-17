from argparse import Namespace, ArgumentParser
from mcq.mcq_distractor_generator import GenerateDistractorsCombineWithAllNoParaphrase
from mcq.mcq_qap_generator import GenerateQuestionAnswerPairs
from mcq.components.distractor_filters import Distractors_Filter
from mcq.components.question_generator import QuestionGenerator
from mcq.components.distractor_generator import DistractorGenerator
from mcq.components.BERT_NER_extractor import NER_extractor
from mcq.components.enums import *
from mcq.mcq_generator import MCQ_Generator
from mcq.components.question_answer_generator import QuestionAnswerGenerator
from kwargs import DG_1_KWARGS, QAG_KWARGS, QG_KWARGS, DG_ALL_KWARGS, PARAPHRASE_KWARGS
from omegaconf import OmegaConf

from transformers import T5ForConditionalGeneration, T5Tokenizer, BertForTokenClassification, BertTokenizer
import json
import os

def parse_argument() -> Namespace:
    
    args = ArgumentParser()
    args.add_argument("--answer_token",type=str, default="<answer>")
    args.add_argument("--context_token", type=str, default="<context>")
    args.add_argument("--question_token", type=str, default="<question>")
    args.add_argument("--sep_token", type=str, default="<sep>")
    args.add_argument("--paraphraser_token", type=str, default="<paraphrase>")
    args.add_argument("--device", type=str, default="cuda")

    args.add_argument("--qg_max_split_sentence", type=int, default=64)
    args.add_argument("--use_evaluator", type=int, default=0, choices=[0, 1])
    args.add_argument("--use_paraphrase", type=int, default=0, choices=[0, 1])
    args.add_argument("--fast_execution", type=int, default=0, choices=[0, 1])
    args.add_argument("--qag_path_base", type=str, default="VosLannack/QAG_ID_Generator_t5_base")
    args.add_argument("--ner_path_base", type=str, default="cahya/bert-base-indonesian-NER")
    args.add_argument("--qg_path_base", type=str, default="VosLannack/QG_ID_Generator_t5_base")
    args.add_argument("--dg_path_base", type=str, default="VosLannack/Distractors_all_base")
    args.add_argument("--qae_path", type=str, default="VosLannack/QAG_ID_Evaluator")
    args.add_argument("--dg_1_path_base", type=str, default="VosLannack/Distractor_1_base_cc")
    args.add_argument("--experiment_type",
                    required=True, help="Check enums file in components folder")

    return args.parse_args()

def execute_(question_json_path: str,
            experiment_qg: ExperimentQG,
            experiment_dg: ExperimentDG,
            mcq: MCQ_Generator,
            result_filtered_name: str,
            result_raw_name: str,
             **kwargs ):
    result = []
    rawResult = []
    with open(question_json_path, "r", encoding="utf-8") as fichier:
        data = json.load(fichier)
    
    for indx, d in enumerate(data):
        ques, all_outputs = mcq(d[f"question_{indx + 1}"], experiment_qg, experiment_dg, **kwargs)
        result.append(ques)
        rawResult.append(all_outputs)
    
    with open(os.path.join("./mcq/mcq_file", result_filtered_name), "w", encoding="utf-8") as fichier:
        json.dump(result, fichier, ensure_ascii=False, indent=4)
    
    with open(os.path.join("./mcq/mcq_file/", result_raw_name), "w", encoding="utf-8") as fichier:
        json.dump(rawResult, fichier, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Use your access token here
    args = parse_argument()
    # Move to document reader for .env token
    # 
    config = OmegaConf.load("config.yaml")
    token = config.h_token

    qag_model = T5ForConditionalGeneration.from_pretrained(args.qag_path_base, use_auth_token=token).to(args.device)
    qag_tokenizer = T5Tokenizer.from_pretrained(args.qag_path_base, use_auth_token=token)

    qg_model = T5ForConditionalGeneration.from_pretrained(args.qg_path_base, use_auth_token=token).to(args.device)
    qg_tokenizer = T5Tokenizer.from_pretrained(args.qg_path_base, use_auth_token=token)

    dg_model = T5ForConditionalGeneration.from_pretrained(args.dg_path_base, use_auth_token=token).to(args.device)
    dg_tokenizer = T5Tokenizer.from_pretrained(args.dg_path_base, use_auth_token=token)

    dg_1_model = T5ForConditionalGeneration.from_pretrained(args.dg_1_path_base, use_auth_token=token).to(args.device)
    dg_1_tokenizer = T5Tokenizer.from_pretrained(args.dg_1_path_base, use_auth_token=token)

    ner_model = BertForTokenClassification.from_pretrained(args.ner_path_base)
    ner_tokenizer = BertTokenizer.from_pretrained(args.ner_path_base)
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

    ds = Distractors_Filter("./embeddings/500k_embeddings.pkl")

    DG_generator = GenerateDistractorsCombineWithAllNoParaphrase(
        distractorPipeline=dg_1_pipeline,
        distractorAllPipeline=dg_all_pipeline,
        distractor_filters=ds,
    )
    mcq = MCQ_Generator(
        QG_generator,
        DG_generator
    )

    kwargs = {
        "kwargs_qg": QG_KWARGS,
        "kwargs_qag":QAG_KWARGS,
        "kwargs_distractor_all": DG_ALL_KWARGS,
        "kwargs_distractor_1": DG_1_KWARGS,
        "kwargs_paraphrase": PARAPHRASE_KWARGS
    }
    path = "./mcq/mcq_file/question_test.json"
    if ExperimentState.QG_DG.value == args.experiment_type:
        execute_(path, ExperimentQG.QG_ONLY.value, ExperimentDG.DG_ONLY.value, mcq, "QG_DG.json", "QG_DG_RAW.json") 

    elif ExperimentState.QG_DAG.value == args.experiment_type:
        execute_(path, ExperimentQG.QG_ONLY.value, ExperimentDG.DAG_ONLY.value, mcq, "QG_DAG.json", "QG_DAG_RAW.json") 

    elif ExperimentState.QAG_DG.value == args.experiment_type:
        execute_(path, ExperimentQG.QAG_ONLY.value, ExperimentDG.DG_ONLY.value, mcq, "QAG_DG.json", "QAG_DG_RAW.json") 

    elif ExperimentState.QAG_DAG.value == args.experiment_type:
        execute_(path, ExperimentQG.QAG_ONLY.value, ExperimentDG.DAG_ONLY.value, mcq, "QAG_DAG.json", "QAG_DAG_RAW.json") 
    
    elif ExperimentState.QG_QAG_DAG.value == args.experiment_type:
        execute_(path, ExperimentQG.QG_QAG.value, ExperimentDG.DAG_ONLY.value, mcq, "QG_QAG_DAG.json", "QG_QAG_DAG_RAW.json") 

    elif ExperimentState.QG_QAG_DG.value == args.experiment_type:
        execute_(path, ExperimentQG.QG_QAG.value, ExperimentDG.DG_ONLY.value, mcq, "QG_QAG_DG.json", "QG_QAG_DG_RAW.json") 

    elif ExperimentState.QG_DG_DAG.value == args.experiment_type:
        execute_(path, ExperimentQG.QG_ONLY.value, ExperimentDG.DG_DAG.value, mcq, "QG_DG_DAG.json", "QG_DG_DAG_RAW.json") 
    
    elif ExperimentState.QAG_DG_DAG.value == args.experiment_type:
        execute_(path, ExperimentQG.QAG_ONLY.value, ExperimentDG.DG_DAG.value, mcq, "QAG_DG_DAG.json", "QAG_DG_DAG_RAW.json") 

    elif ExperimentState.QG_QAG_DG_DAG.value == args.experiment_type:
        execute_(path, ExperimentQG.QG_QAG.value, ExperimentDG.DG_DAG.value, mcq, "QG_QAG_DG_DAG.json", "QG_QAG_DG_DAG_RAW.json") 

    #ques = mcq(context, **kwargs)
    #print(ques)
    #distractors = DG_generator(question, context, answer, **kwargs)
    #print(distractors)
    
    #print("question", question)
    #print("Jawaban benar", answer)
    #rint("Pengecoh",distractors)
    #pairs = QG_generator.generate_question_from_QAG(context)
    #print(pairs)