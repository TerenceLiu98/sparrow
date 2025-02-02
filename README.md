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

## Data Preparation

1. For tokenizer and pretraining process, to simplify the data collection process, we use the data from [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), ensuring that our training corpus is both rich in content and easily accessible. We use 10%-20% of the data with six official language of United Nation — Arabic, Chinese, English, French, Russian, and Spanish—providing a diverse and representative sample for training our tokenizer.

## Tokenizer

A good tokenizer is vital as it is the first component that converts raw text into a structured format a model can understand. It determines the granularity of tokenization and ensures that important elements—such as special tokens marking the beginning and end of a sentence—are consistently incorporated, directly affecting the model's ability to learn and generate language accurately. In `tokenizer/tokenizer.py`, we provide a `class SparrowTokenizer` to help you understand the how a tokenizer been trained. This script demonstrates the complete pipeline—from preprocessing raw data and creating a training corpus, to training a BPE-based tokenizer with customized post-processing for adding special tokens, and finally, saving the vocabulary and configuration files. You can explore this workflow by running:

```bash
python tokenizer/tokenizer.py --args configs/tokenizers.yaml
```