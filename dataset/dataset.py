import json
import numpy as np

import torch
from torch.utils.data import IterableDataset, Dataset

class PretrainDataset(IterableDataset):
    '''
    For the pretraining process, we are doing the causal language modelling task (unsupervised):
        1. assume all data are `content`:
        2. tokenize the `content` into `input_ids` with bos and eos
        3. for CLM task, `input_ids` = `label_ids`
    '''
    def __init__(self, data_path, tokenizer, max_seq_len):
        super(PretrainDataset, self).__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __iter__(self):
        return self.data_process()
    
    def data_process(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                text = '<s>' + line['text'] + '</s>'
                input_ids = self.tokenizer.encode(text)
                text_len = len(input_ids)
                if text_len > self.max_seq_len:
                    input_ids = input_ids[:self.max_seq_len]
                else:
                    input_ids = input_ids + [-100] * (self.max_seq_len - text_len)
                input_ids = np.array(input_ids)
                X = np.array(input_ids[:-1]).astype(np.int64)
                Y = np.array(input_ids[1:]).astype(np.int64)
                yield {
                    'input_ids': torch.from_numpy(X),
                    'output_ids': torch.from_numpy(Y),
                }

class InstructDataset(IterableDataset):
    '''
    For the instruct tuning process, we are dogin the supervised Fine-tuning task (supervised):
        1. the dataset should instruct three features: `instruct`, `input`, `output`
        2. prompt = `instruct` + `input` and output = `output`
        3. tokenize both prompt and output into `input_ids` and `output_ids`
        4. We need to concatenate the `input_ids` and the `output_ids`, 
            however, the concatenated `input_ids`'s value should be subsituted by [-100] (as -100 is typically used in CrossEntopyLoss as ignore_index)
    '''
    def __init__(self, data_path, tokenizer, max_seq_len):
        super(InstructDataset, self).__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __iter__(self):
        return self.data_process()
    
    def format_func(self, example):
        message, output = [], []
        if example["instruct"]:
            message.append({"role": "system", "content": example["instruct"]})
        if example["input"]:
            message.append({"role": "user", "content": example["input"]})
        output.append({"role": "assistant", "content": example["output"]})
        return message, output
    
    def data_process(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                message, output = self.format_func(example)
                input_ids = self.tokenizer.apply_chat_template(message)
                output_ids = self.tokenizer.apply_chat_template(output)
                output_ids = [-100] * len(input_ids) + output_ids
                
                if len(input_ids) > self.max_seq_len:
                    input_ids = input_ids[:self.max_seq_len]
                    output_ids = output_ids[:self.max_seq_len]
                else:
                    input_ids = input_ids + [self.tokenizer.eos_token_id] * (self.max_seq_len - len(input_ids))
                    output_ids = output_ids[:self.max_seq_len] if len(output_ids) > self.max_seq_len else \
                        output_ids + [self.tokenizer.eos_token_id] * (self.max_seq_len - len(output_ids))
                
                yield {
                    'input_ids': torch.tensor(input_ids), 
                    'labels': torch.tensor(output_ids),
                }

if __name__ == "__main__":
    data_path = "/data/sparrow/train.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("/data/sparrow/tokenizer")
    max_seq_len = 512
    dataset = PretrainDataset(data_path=data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)
    dataset = InstructDataset(data_path="/data/sparrow/data/valid.jsonl", tokenizer=tokenizer, max_seq_len=max_seq_len)