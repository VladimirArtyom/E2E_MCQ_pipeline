from argparse import Namespace, ArgumentParser
from mcq.mcq_distractor_generator import GenerateDistractorsCombineWithAllNoParaphrase, GenerateDistractorParaphrase
from mcq.mcq_qap_generator import GenerateQuestionAnswerPairs
from mcq.components.distractor_filters import Distractors_Filter
from mcq.components.question_generator import QuestionGenerator
from mcq.components.distractor_generator import DistractorGenerator
from mcq.components.BERT_NER_extractor import NER_extractor
from mcq.components.paraphrase_question import ParaphraseQuestion
from mcq.mcq_generator import MCQ_Generator
from mcq.components.question_answer_generator import QuestionAnswerGenerator
from kwargs import DG_1_KWARGS, QAG_KWARGS, QG_KWARGS, DG_ALL_KWARGS, PARAPHRASE_KWARGS
from omegaconf import OmegaConf

from pickle import dump
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertForTokenClassification, BertTokenizer
import json

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
    args.add_argument("--dg_path_base", type=str, default="VosLannack/Distractor_all_base_cc")
    args.add_argument("--qae_path", type=str, default="VosLannack/QAG_ID_Evaluator")
    args.add_argument("--dg_1_path_base", type=str, default="VosLannack/Distractor_1_base_cc")
    args.add_argument("--paraphrase_path", type=str, default="Wikidepia/IndoT5-base-paraphrase")

    return args.parse_args()

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

    #qae_model = BertForSequenceClassification.from_pretrained(args.qae_path, use_auth_token=token)
    #qae_tokenizer = AutoTokenizer.from_pretrained(args.qae_path, use_auth_token=token)

    dg_model = T5ForConditionalGeneration.from_pretrained(args.dg_path_base, use_auth_token=token).to(args.device)
    dg_tokenizer = T5Tokenizer.from_pretrained(args.dg_path_base, use_auth_token=token)

    dg_1_model = T5ForConditionalGeneration.from_pretrained(args.dg_1_path_base, use_auth_token=token).to(args.device)
    dg_1_tokenizer = T5Tokenizer.from_pretrained(args.dg_1_path_base, use_auth_token=token)

    ner_model = BertForTokenClassification.from_pretrained(args.ner_path_base)
    ner_tokenizer = BertTokenizer.from_pretrained(args.ner_path_base)
    ner = NER_extractor(ner_model, ner_tokenizer)

    paraphrase_model = T5ForConditionalGeneration.from_pretrained(args.paraphrase_path).to(args.device)
    paraphrase_tokenizer = T5Tokenizer.from_pretrained(args.paraphrase_path)

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
    paraphrase_pipeline = ParaphraseQuestion(
        paraphrase_model,
        paraphrase_tokenizer,
        512,
        args.device
    )

    
    ds = Distractors_Filter("./embeddings/500k_embeddings.pkl")
    #DG_generator = GenerateDistractorParaphrase(
    #    distractorPipeline=dg_1_pipeline,
    #    paraphrasePipeline=paraphrase_pipeline,
    #    distractor_filters=ds
    #)
    DG_generator = GenerateDistractorsCombineWithAllNoParaphrase(
        distractorPipeline=dg_1_pipeline,
        distractorAllPipeline=dg_all_pipeline,
        distractor_filters=ds
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
    with open("./mcq/mcq_file/questions.json", "r", encoding="utf-8") as fichier:
        data = json.load(fichier)

    result = []
    for indx, d in enumerate(data):
        ques, all_outputs = mcq(d[f"question_{indx + 1}"], **kwargs)
        result.append(ques)

    with open("./mcq/mcq_file/result.json", "w", encoding="utf-8") as fichier:
        json.dump(result, fichier, ensure_ascii=False, indent=4)
    
    with open("./mcq/mcq_file/result_raw.pickle", "wb") as fichier:
        dump(all_outputs, fichier)

    #ques = mcq(context, **kwargs)
    #print(ques)
    #distractors = DG_generator(question, context, answer, **kwargs)
    #print(distractors)
    
    #print("question", question)
    #print("Jawaban benar", answer)
    #rint("Pengecoh",distractors)
    #pairs = QG_generator.generate_question_from_QAG(context)
    #print(pairs)