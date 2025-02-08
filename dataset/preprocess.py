import numpy as np
from tqdm import tqdm
from datasets import concatenate_datasets, load_dataset, Features, Value

class DataPreprocess(object):
    def __init__(self, dataset_name: str="wikimedia/wikipedia", 
                        dataset_subset: list=None, 
                        dataset_size: float=1.2e6,
                        path: str="/data/sparrow/"):
        super().__init__()
    
        self.dataset_name = dataset_name 
        self.dataset_subset = dataset_subset
        self.dataset_size = dataset_size
        self.path = path
    
    def pretrain_dataset(self):
        num_row = np.arange(0, self.dataset_size // len(self.dataset_subset), dtype=int).tolist()
        data_list = [load_dataset(self.dataset_name, data_dir=self.dataset_subset[i],\
            split="train").shuffle().select(num_row) for i in tqdm(range(0, len(self.dataset_subset)))]
        
        self.dataset = concatenate_datasets(data_list).shuffle()
        self.dataset.to_json(str(self.path + "pretrain.jsonl"))
        print("[:)] pretrain dataset saved")
    
if __name__ == "__main__":
    ps = DataPreprocess(dataset_name="wikimedia/wikipedia", 
        dataset_subset=['20231101.ru', '20231101.zh', '20231101.fr', '20231101.es', '20231101.en', '20231101.ar'])
    ps.pretrain_dataset()

    dolphin = load_dataset("cognitivecomputations/dolphin", "flan5m-alpaca-uncensored", split="train")
    dolphin = dolphin.rename_column("instruction", "instruct")
    openorca = load_dataset("Open-Orca/OpenOrca", split="train")
    openorca = openorca.remove_columns(["id"])
    openorca = openorca.rename_column("system_prompt", "instruct")
    openorca = openorca.rename_column("question", "input")
    openorca = openorca.rename_column("response", "output")
    dataset = concatenate_datasets([dolphin, openorca]).shuffle()
    dataset = dataset.train_test_split(0.05)
    trainset, validset = dataset["train"], dataset["test"]
    trainset.to_json(ps.path + "train.jsonl")
    validset.to_json(ps.path + "valid.jsonl")