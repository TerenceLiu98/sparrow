<div align="center">

# Sparrow

麻雀虽小 五脏俱全

 Small as it is, the sparrow has all the vital organs

<image src=".github/sparrow.png" width="300" />

</div>


The **sparrow** project aims to help beginner to understand the base architecture of a large language model from scratch. Not only the model, but also the optimization methods that are widely use to shorten the training process.

- [x] tokenizer from scratch & merge tokenizer
- [x] base model modules from scratch & train the stacked model
- [ ] supervised fine-tuning
- [ ] Reward Modelling
- [ ] Mixture-of-Expert

## Before

1. For tokenizer and pretraining process, to simplify the data collection process, we use the data from [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), ensuring that our training corpus is both rich in content and easily accessible. We use 0.2M of the data for each six official languages of United Nation — Arabic, Chinese, English, French, Russian, and Spanish—providing a diverse and representative sample for training our tokenizer.
2. For pretrain process, we also use [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) to ensure the model's token embedding and tokenizer's alignment. 
3. For the instruct tuning process, we adopt the [`Open-Orca/OpenOrca`](https://huggingface.co/datasets/Open-Orca/OpenOrca) and [`cognitivecomputations/dolphin`](https://huggingface.co/datasets/cognitivecomputations/dolphin) as the dataset. (Optional: [`kaist-ai/CoT-Collection`](https://huggingface.co/datasets/kaist-ai/CoT-Collection)) 
4. All parameters of training are located in `./configs/`, you may change it before you train each module

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
from transformers import Trainer, TrainingArguments, DefaultDataCollator, AutoTokenizer, AutoModelForCausalLM, AutoConfig

device = "cuda:0"

# register model and config before use
AutoConfig.register("sparrow", SparrowConfig)
AutoModelForCausalLM.register(SparrowConfig, SparrowModelForCausalLM)
model = AutoModelForCausalLM.from_pretrained("{model_location}").to(device)
tokenizer = AutoTokenizer.from_pretrained("{tokenizer_location}")

input_ids = tokenizer.encode("<s>United Nation is",  add_special_tokens=False, return_tensors="pt").to(model.device)
output_ids = model.generate(input_ids, eos=tokenizer.eos_token_id, max_new_tokens=30, \
    temperature=0.7, top_k=5, use_beam_search=True, beam_size=3)
print(f"generated output: {tokenizer.decode(output_ids, skip_special_tokens=False)}")
```