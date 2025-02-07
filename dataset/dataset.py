import json
import numpy as np

import torch
from torch.utils.data import IterableDataset, Dataset

class PretrainDataset(IterableDataset):
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
                    input_ids = input_ids + [0] * (self.max_seq_len - text_len)
                input_ids = np.array(input_ids)
                X = np.array(input_ids[:-1]).astype(np.int64)
                Y = np.array(input_ids[1:]).astype(np.int64)
                yield {
                    'input_ids': torch.from_numpy(X),
                    'labels': torch.from_numpy(Y),
                }

class InstructDataset(IterableDataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super(InstructDataset, self).__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __iter__(self):
        return self.data_process()
    
    def format_func(self, example):
        messages = []
        if example["instruct"]:
            messages.append({"role": "system", "content": example["instruct"]})
        if example["input"]:
            messages.append({"role": "user", "content": example["input"]})
        messages.append({"role": "assistant", "content": example["output"]})
        return messages
    
    def data_process(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                message = format_func(line)
                tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=False)
                input_ids = tokenized["input_ids"].squeeze(0)
                labels = input_ids.clone()
                system_length = len(tokenizer.apply_chat_template([{"role": "system", "content": example["instruct"]}], add_generation_prompt=False)) if example["instruct"] else 0
                user_length = len(tokenizer.apply_chat_template([{"role": "user", "content": example["input"]}], add_generation_prompt=False)) if example["input"] else 0
                assistant_start = system_length + user_length
                labels[:assistant_start] = -100  
                yield {
                    'input_ids': input_ids, 
                    'labels': labels
                }


if __name__ == "__main__":
    data_path = "/data/sparrow/train.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("/data/sparrow/tokenizer")
    max_seq_len = 512
    dataset = PretrainDataset(data_path=data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)