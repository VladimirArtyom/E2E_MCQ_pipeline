from argparse import Namespace, ArgumentParser

def parse_argument() -> Namespace:
    
    args = ArgumentParser()
    args.add_argument("--answer_token",type=str, default="<answer>")
    args.add_argument("--context_token", type=str, default="<context>")
    args.add_argument("--question_token", type=str, default="<question>")
    args.add_argument("--paraphraser_token", type=str, default="<paraphrase>")

    args.add_argument("--qg_max_split_sentence", type=int, default=64)
    args.add_argument("--qae_use_evaluator")
    args.add_argument("--")

    return args.parse_args()



if __name__:
    args = parse_argument()