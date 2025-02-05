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
    

if __name__ == "__main__":
    data_path = "/data/sparrow/train.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("/data/sparrow/tokenizer")
    max_seq_len = 512
    dataset = PretrainDataset(data_path=data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)