from mcq.mcq_distractor_generator import GenerateDistractorParaphrase
from mcq.mcq_qap_generator import GenerateQuestionAnswerPairs
"""
class Executor():
    def __init__(this,
                 args: Namespace,
                 use_evaluator: bool = 0,
                 use_paraphrase: bool = 0,
                 fast_execution: bool = 0,
                 ):
        
        this.use_evaluator = use_evaluator
        this.use_paraphrase = use_paraphrase
        this.fast_execution = fast_execution
        this.args = args

        this.evaluator_path = None
        this.paraphrase_path = None

        this.qg_path = None
        this.qag_path = None
        this.dg_path = None
        this.dg_1_path = None
    
    def prepare_qag_pipeline(this):
        qag_model = T5ForConditionalGeneration.from_pretrained(this.qag_path, use_auth_token=token)
        qag_tokenizer = T5Tokenizer.from_pretrained(this.qag_path, use_auth_token=token)
        this.qag_pipeline = QuestionAnswerGenerator(qag_model,
                                                    qag_tokenizer,
                                                    device=args.device,
                                                    max_length=512,
                                                    context_token=args.context_token)
    
    def prepare_qg_pipeline(this):
        qg_model = T5ForConditionalGeneration.from_pretrained(this.qg_path, use_auth_token=token)
        qg_tokenizer = T5Tokenizer.from_pretrained(this.qg_path, use_auth_token=token)
        this.qg_pipeline = QuestionGenerator(
            qg_model,
            qg_tokenizer,
            device=this.args.device,
            answer_token=this.args.answer_token,
            context_token=this.args.context_token,
            max_length=512
        )
    
    def prepare_distractor_pipeline(this):
        dg_model = T5ForConditionalGeneration.from_pretrained(this.dg_path, use_auth_token=token)
        dg_tokenizer = T5Tokenizer.from_pretrained(this.dg_path, use_auth_token=token)
        this.dg_pipeline = DistractorGenerator(
            dg_model,
            dg_tokenizer,
            512,
            question_token=this.args.question_token,
            context_token=this.args.context_token,
            answer_token=this.args.answer_token,
            sep_token=this.args.sep_token,
            device=this.args.device
        )
    
    def prepare_qae_pipeline(this):
        qae_model = BertForSequenceClassification()
        qae_model = BertForSequenceClassification.from_pretrained(args.qae_path, use_auth_token=token)
        qae_tokenizer = BertTokenizer.from_pretrained(args.qae_path, use_auth_token=token)
        this.qae_pipeline = QuestionAnswerEvaluator(
            qae_model,
            qae_tokenizer,
            512,
            args.device
        )

    def prepare_paraphrase_pipeline(this):
        paraphrase_model = T5ForConditionalGeneration.from_pretrained(this.paraphrase_path)
        paraphrase_tokenizer = T5Tokenizer.from_pretrained(this.paraphrase_path)
        this.paraphrase_pipeline = ParaphraseQuestion(
            paraphrase_model,
            paraphrase_tokenizer,
            512,
            args.device
        )      

    def prepare_distractor_1_pipeline(this):
        dg_1_model = T5ForConditionalGeneration.from_pretrained(this.dg_1_path, use_auth_token=token)
        dg_1_tokenizer = T5Tokenizer.from_pretrained(this.dg_1_path, use_auth_token=token)
        this.dg_1_pipeline = DistractorGenerator(
            dg_1_model,
            dg_1_tokenizer,
            128,
            question_token=this.args.question_token,
            context_token=this.args.context_token,
            answer_token=this.args.answer_token,
            sep_token=this.args.sep_token,
            device=this.args.device
        )

    def prepare_ner_pipeline(this):
        ner_model = BertForTokenClassification.from_pretrained(this.ner_path)
        ner_tokenizer = BertTokenizer.from_pretrained(this.ner_path)
        this.ner = NER_extractor(ner_model, ner_tokenizer)

    def prepare_model(this):
        if this.use_evaluator:
            this.evaluator_path = this.args.qae_path
        
        if this.use_paraphrase:
            this.paraphrase_path = this.args.paraphrase_path
        
        if this.fast_execution:
            this.qg_path = this.args.qg_path_small
            # Currently use the base for QAG, i forgot to trained it on small
            this.qag_path = this.args.qag_path_base
            this.dg_path = this.args.dg_path_small
            this.dg_1_path = this.args.dg_path_1_small
        else:
            this.qg_path = this.args.qg_path_base
            this.qag_path = this.args.qag_path_base
            this.dg_path = this.args.dg_path_base
            ## Currently using the small for single distractors
            this.dg_1_path = this.args.dg_path_1_base

        if this.evaluator_path:
            this.prepare_qae_pipeline()

        if this.paraphrase_path:
            this.prepare_paraphrase_pipeline()

        this.prepare_qag_pipeline()
        this.prepare_qg_pipeline()
        this.prepare_distractor_pipeline()
        this.prepare_distractor_1_pipeline()
        this.prepare_ner_pipeline()

"""
class MCQ_Generator():
    def __init__(this,
                 qg_generator: GenerateQuestionAnswerPairs,
                 dg_generator: GenerateDistractorParaphrase,
                 ):
        this.qg_generator = qg_generator
        this.dg_generator = dg_generator
    
    def __call__(this, context: str, **kwargs):
        final_outputs = {}
        list_de_qag = this.qg_generator(context, **kwargs)
        for indx, content in enumerate(list_de_qag):
            question = content[0]
            answer = content[1]
            distractors, all_outputs_raw = this.dg_generator(context, question, answer, **kwargs)
            final_outputs[indx] = {
                "context": context,
                "question": question,
                "answer": answer,
                "distractors": distractors
            }
        return final_outputs, all_outputs_raw