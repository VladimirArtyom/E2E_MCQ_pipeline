from argparse import Namespace, ArgumentParser
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration
import pandas as pd 
from kwargs import QG_KWARGS, QAG_KWARGS, DG_ALL_KWARGS, DG_1_KWARGS
from question_answer_generator import QuestionAnswerGenerator
from question_generator import QuestionGenerator
from distractor_generator import DistractorGenerator
from omegaconf import OmegaConf
from typing import Mapping
import datasets

def parse_args() -> Namespace:
    args = ArgumentParser()
    args.add_argument("--model", type=str, required=True, help="HuggingFace Path model")
    args.add_argument("--model_type", type=str, required=True, choices=["qg", "qag", "dg", "dg_1"])
    args.add_argument("--file_path", type=str, required=True)
    args.add_argument("--save_file_name", type=str, required=True)
    args.add_argument("--eval_type", type=str, required=True, choices=["val", "test"], default="val")
    args.add_argument("--context_token", type=str, default="<context>")
    args.add_argument("--answer_token", type=str, default="<answer>")
    args.add_argument("--sep_token", type=str, default="<sep>")
    args.add_argument("--question_token", type=str, default="<question>")
    args.add_argument("--device", type=str, default="cpu")
    args.add_argument("--max_length", type=int, default=512)

    return args.parse_args()

def predict_with_qag(qag: QuestionAnswerGenerator, data: pd.DataFrame):
    new_dataFrame = pd.DataFrame(columns=["question", "answer", "context"])
    for indx, item in data.iterrows():
        pred = qag(item["context"], **QAG_KWARGS)
        new_observation: Mapping = {
            "indx": indx,
            "question": item["question"],
            "answer": item["answer"],
            "pred_qag": pred,
            "context": item["context"],
        }
        if indx % 50 == 0 and indx != 0:
            print("Currently running on {} out of {}".format(indx, len(indx)))
        new_dataFrame = pd.concat([new_dataFrame, pd.DataFrame(new_observation)], axis=0, ignore_index=True)
        if indx == 4 :
            break
        
    
    return new_dataFrame

def predict_with_qg(qg: QuestionGenerator, data: pd.DataFrame ):
    new_dataFrame = pd.DataFrame(columns=["question", "answer", "context"])
    for indx, item in data.iterrows():
        context = item["context"]
        answer = item["answer"]
        question = item["question"]
        pred = qg(answer=answer, context=context, **QG_KWARGS)
        new_observation: Mapping = {
            "indx": indx,
            "question": question,
            "answer": answer,
            "pred_qg": pred,
            "context": context,
        }
        if indx % 50 == 0 and indx != 0:
            print("Currently running on {} out of {}".format(indx, len(indx)))
        
        new_dataFrame = pd.concat([new_dataFrame, pd.DataFrame(new_observation)], axis=0, ignore_index=True)
        if indx == 4 :
            break
        
    return new_dataFrame

def predict_with_dg(dg: DistractorGenerator, data: pd.DataFrame):
    new_dataFrame = pd.DataFrame(columns=["question", "correct", "context", "incorrect_1", "incorrect_2", "incorrect_3", "pred_incorrects"],)
    for indx, item in data.iterrows():
        context = item["context"]
        answer = item["correct"]
        question = item["question"]
        incorrect_1 = item["incorrect_1"]
        incorrect_2 = item["incorrect_2"]
        incorrect_3 = item["incorrect_3"]
        pred = dg(question=question, context=context, answer=answer, **DG_ALL_KWARGS)
        new_observation: Mapping = {
            "question": question,
            "correct": answer,
            "context": context,
            "incorrect_1": incorrect_1,
            "incorrect_2": incorrect_2,
            "incorrect_3": incorrect_3,
            "pred_incorrects": pred,
        }
        if indx % 50 == 0 and indx != 0:
            print("Currently running on {} out of {}".format(indx, len(indx)))
        new_dataFrame = pd.concat([new_dataFrame, pd.DataFrame(new_observation)], axis=0, ignore_index=True)
        if indx == 4 :
            break
        
    return new_dataFrame

def predict_with_dg_1(dg: DistractorGenerator, data: pd.DataFrame):
    new_dataFrame = pd.DataFrame(columns=["question", "correct", "context", "incorrect", "pred_incorrect"],)
    for indx, item in data.iterrows():
        context = item["context"]
        answer = item["correct"]
        question = item["question"]
        incorrect= item["incorrect"]
        pred = dg(question=question, context=context, answer=answer, **DG_1_KWARGS)
        new_observation: Mapping = {
            "question": question,
            "correct": answer,
            "context": context,
            "incorrect": incorrect,
            "pred_incorrect": pred,
        }
        if indx % 50 == 0 and indx != 0:
            print("Currently running on {} out of {}".format(indx, len(indx)))
        new_dataFrame = pd.concat([new_dataFrame, pd.DataFrame(new_observation)], axis=0, ignore_index=True)
        if indx == 4 :
            break
    return new_dataFrame

if __name__ == "__main__":
    args: Namespace = parse_args()
    config = OmegaConf.load("config.yaml")

    model = T5ForConditionalGeneration.from_pretrained(args.model, use_auth_token=config.h_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_auth_token=config.h_token)
    data = datasets.load_dataset(args.file_path)
    val = pd.DataFrame(data["validation"])
    test = pd.DataFrame(data["test"])
    if args.model_type == "qg":
        qg = QuestionGenerator(
            model,
            tokenizer,
            answer_token=args.answer_token,
            context_token=args.context_token,
            max_length=512,
            device=args.device
        )
        if args.eval_type == "val":
           df_new = predict_with_qg(qg, val)
        else:
           df_new = predict_with_qg(qg, test)
    elif args.model_type == "qag":
        qag = QuestionAnswerGenerator(
            model,
            tokenizer,
            args.max_length,
            args.context_token,
            args.device
        )
        if args.eval_type == "val":
           df_new = predict_with_qag(qag, val)
        else:
           df_new = predict_with_qag(qag, test)
    elif args.model_type == "dg":
        dg = DistractorGenerator(
            model,tokenizer,512,
            question_token=args.question_token,
            context_token=args.context_token,
            answer_token=args.answer_token,
            sep_token=args.sep_token,
            device=args.device
        )
        if args.eval_type == "val":
           df_new = predict_with_dg(dg, val)
        else:
           df_new = predict_with_dg(dg, test)
    elif args.model_type == "dg_1":
        dg = DistractorGenerator(
            model,tokenizer,512,
            question_token=args.question_token,
            context_token=args.context_token,
            answer_token=args.answer_token,
            sep_token=args.sep_token,
            device=args.device
        )
        if args.eval_type == "val":
           df_new = predict_with_dg_1(dg, val)
        else:
           df_new = predict_with_dg_1(dg, test)

    df_new.to_csv(args.save_file_name)
    
