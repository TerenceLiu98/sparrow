<div align="center">

# Sparrow

麻雀虽小 五脏俱全

 Small as it is, the sparrow has all the vital organs

<image src=".github/sparrow.png" width="300" />

</div>


The **sparrow** project aims to help beginner to understand the base architecture of a large language model from scratch. Not only the model, but also the optimization methods that are widely use to shorten the training process.

- [ ] tokenizer from scratch & merge tokenizer
- [ ] model modules from scratch & train the stacked model
- [ ] supervised fine-tuning
- [ ] Reward Modelling

## Before

1. For tokenizer and pretraining process, to simplify the data collection process, we use the data from [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), ensuring that our training corpus is both rich in content and easily accessible. We use 10%-20% of the data with six official language of United Nation — Arabic, Chinese, English, French, Russian, and Spanish—providing a diverse and representative sample for training our tokenizer.
2. All parameters of training are located in `./configs/`, you may change it before you train each module

## Tokenizer

A good tokenizer is vital as it is the first component that converts raw text into a structured format a model can understand. It determines the granularity of tokenization and ensures that important elements—such as special tokens marking the beginning and end of a sentence—are consistently incorporated, directly affecting the model's ability to learn and generate language accurately. In `tokenizer/tokenizer.py`, we provide a `class SparrowTokenizer` to help you understand the how a tokenizer been trained. This script demonstrates the complete pipeline—from preprocessing raw data and creating a training corpus, to training a BPE-based tokenizer with customized post-processing for adding special tokens, and finally, saving the vocabulary and configuration files. You can explore this workflow by running:

```bash
bash a_tokenizer.sh
```

## Models Artitecture 

skip

## Pretrain Process

Based on the model, we can build a training process based on huggingface's [`TrainingArgument`](https://huggingface.co/docs/transformers/v4.48.0/en/main_classes/trainer#transformers.TrainingArguments) and [`Trainer`](https://huggingface.co/docs/transformers/v4.48.0/en/main_classes/trainer#transformers.Trainer). You can explore this workflow by running:

```base
bash b_pretrain.sh
```

To evaluate the trained model:

```python
from dataset.dataset import *
from model.modelling_sparrow import * 
from model.configuration_sparrow import * 
from transformers import Trainer, TrainingArguments, DefaultDataCollator, AutoTokenizer, AutoModelForCausalLM, AutoConfig

device = "cuda:0"

# register model and config before use
AutoConfig.register("sparrow", SparrowConfig)
AutoModelForCausalLM.register(SparrowConfig, SparrowModel)
model = AutoModelForCausalLM.from_pretrained("{model_location}").to(device)
tokenizer = AutoTokenizer.from_pretrained("{tokenizer_location}")

input_ids = tokenizer.encode("<s>United Nation is",  add_special_tokens=False, return_tensors="pt").to(model.device)
for token in model.generate(input_ids, tokenizer.eos_token_id, 30, stream=False,temperature=0.7, top_k=5):
    print(tokenizer.decode(token[0]))
```