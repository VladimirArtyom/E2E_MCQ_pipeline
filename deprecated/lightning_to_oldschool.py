from argparse import Namespace, ArgumentParser
from torch import load
from transformers import T5ForConditionalGeneration, AutoTokenizer, AdamW
from torch import Tensor
from pytorch_lightning import LightningModule
from typing import Dict
from huggingface_hub import HfApi, HfFolder
import torch
def parser() -> Namespace:
    args = ArgumentParser()
    args.add_argument("--file_ckpt", required=True, type=str)

class DGModel(LightningModule):
    def __init__ (this, 
                  model: T5ForConditionalGeneration,
                  new_tokenizer_len: int,
                  optimizer,
                  optimizer_lr: float = 1e-4):
        super().__init__()
        this.model = model
        this.model.resize_token_embeddings(new_tokenizer_len)
        this.lr = optimizer_lr
        this.opt = optimizer

    def forward(this, input_ids, attention_mask, labels=None):
        output: Tensor = this.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
        return output.loss, output.logits

    def training_step(this, batch: Dict, batch_indx: int):
        loss = this.exe_step(batch, batch_indx)
        this.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(this, batch: Dict, batch_indx: int):
        loss = this.exe_step(batch, batch_indx)
        this.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(this, batch: Dict, batch_indx: int):
        loss = this.exe_step(batch, batch_indx)
        this.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def exe_step(this, batch: Dict, batch_indx: int):
        input_ids: Tensor = batch["input_ids"]
        attention_mask: Tensor = batch["attention_mask"]
        labels: Tensor = batch["labels"]
        loss, output = this(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        return loss

    def configure_optimizers(this):
        return this.opt(this.parameters(),
                        lr=this.lr)

if __name__ == "__main__":
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    tokenizer.add_special_tokens({
        "additional_special_tokens" :["<question>", "<answer>", "<context>", "<sep>"]
    })
    model.resize_token_embeddings(len(tokenizer))
    dgModel = DGModel(model, len(tokenizer), AdamW, 1e-4)
    dgModel.load_state_dict(load("./model_thesis/model_distractor_1_base/best-checkpoint_e4.ckpt")["state_dict"])
    print("model_loaded")
    dgModel.model.save_pretrained("model-distractors_1_base")
    tokenizer.save_pretrained("model-distractors_1_base/tokenizer")
    #torch.save(dgModel.state_dict(), "QAG_t5_base/pytorch_model.bin")

    
