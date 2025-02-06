import yaml
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.dataset import *
from model.modelling_sparrow import *
from model.configuration_sparrow import SparrowConfig

from transformers import HfArgumentParser, Trainer, TrainingArguments, AutoTokenizer, DefaultDataCollator

@dataclass
class DataArguments:
    train_path: str=field(default=None, metadata={"help": "Path of the train dataset"}) 
    tokenizer_path: str=field(default="/data/sparrow/tokenizer", metadata={"help": "Path of the tokenizer"})
    model_path: str=field(default="/data/sparrow/model", metadata={"help": "Path of the trained model - finalized"})
    cache_path: str=field(default="/data/sparrow/tmp", metadata={"help": "Path of tmp file of model training (checkpoint)"})

@dataclass
class ModelArguments:
    # Attention parameters
    hidden_size: int = field(default=512)
    num_hidden_layers: int = field(default=8)
    num_attention_heads: int = field(default=16)
    num_key_value_heads: Optional[int] = field(default=None)
    max_seq_len: int = field(default=512)
    attention_bias: bool = field(default=False)
    flash_attn: bool = field(default=True)
    # MLP parameters
    vocab_size: int = field(default=32000)
    hidden_dim: Optional[int] = field(default=None)
    intermediate_dim: int = field(default=2048)
    norm_eps: float = field(default=1e-5)
    mlp_bias: bool = field(default=False)
    dropout: float = field(default=0.0)

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./output")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # optimizer
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

class SparrowForCausalLM(object):
    def __init__(self, data_args, model_args, train_args):
        super(SparrowForCausalLM, self).__init__()
        self.data_args = data_args
        self.train_args = train_args
        self.model_args = SparrowConfig(**model_args.__dict__)
        self.load_model()
        self.load_dataset()
    
    def load_dataset(self):
        self.trainset = PretrainDataset(data_path=self.data_args.train_path, tokenizer=self.tokenizer, max_seq_len=self.train_args.model_max_length)

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.data_args.tokenizer_path)
        self.model = SparrowModel(self.model_args)
    
    def train(self):
        self.trainer  = Trainer(
            model=self.model,
            args=self.train_args,
            train_dataset=self.trainset,
            processing_class=self.tokenizer, 
            data_collator=DefaultDataCollator()
        )

        num_params = sum(p.numel() for p in self.model.parameters()) // 1.0e6
        print(f"The Model has {num_params}M parameters\n")
        self.trainer.train()

    def save(self):
        self.model.save_pretrained(self.data_args.model_path, safe_serialization=False)
        self.tokenizer.save_pretrained(self.data_args.model_path)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="Pretraining Argument YAML")
    args_parser.add_argument("--args", type=str, required=False, default="./params.yaml")
    args_parser = args_parser.parse_args()
    args = yaml.safe_load(Path(f"{args_parser.args}").read_text())

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_dict(args)
    sparrow = SparrowForCausalLM(data_args, model_args, train_args)
    sparrow.train()
    sparrow.save()