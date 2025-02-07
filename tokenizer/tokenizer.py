import os
import json
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from tokenizers import decoders, models, pre_tokenizers, processors, trainers, Tokenizer

from transformers import AutoTokenizer

class SparrowTokenizer:
    """
    params:
        vocab_file (str, optional): A path to a pre-defined vocabulary file that helps initialize or guide the
            training of the token-to-id mapping. If provided, the tokenizer can leverage this vocab instead of
            building one entirely from scratch. Default is None.
        vocab_size (int): The maximum size of the vocabulary. This number defines how many tokens will be kept
            in the final vocabulary. Default is 32000, same as Llama3.
        min_frequency (int): The minimum frequency a token must appear in the dataset to be included in the vocabulary.
            Tokens appearing less frequently than this threshold will be discarded. Default is 2.
        special_tokens (list): A list of special tokens that will be reserved in the vocabulary. These include tokens
            for the unknown tokens (<unk>), beginning-of-sentence (<s>), end-of-sentence (</s>), padding (<pad>), and mask tokens (<mask>).
            The order in this list can determine their fixed positions (indices) in the vocabulary. Default is ["<s>", "</s>", "<pad>", "<unk>", "<mask>"].
        dataset_name (str): The name of the dataset from which the corpus will be loaded for tokenizer training.
            Default is "wikimedia/wikipedia".
        dataset_subset (list): A list of subset identifiers or data directories to specify which parts of the dataset
            should be used for training the tokenizer. Default is ['20231101.ru', '20231101.zh', '20231101.fr',
            '20231101.es', '20231101.en', '20231102.ar'], the six official languages used in the United Nation.
        dataset_ratio (float): How many data are used for tokenizer training. Default is 0.1 (randomly).
        remove_columns (list): A list of column names to be removed from the dataset during preprocessing.
            This is typically used to eliminate metadata columns that are not needed for tokenization. Default is ["id", "url", "title"].
        tokenizer_path (str): The directory path where the tokenizer artifacts (e.g., trained vocabulary, merges file, and any
            generated configuration files) will be saved or loaded from. Default is "./".
    """
    def __init__(
        self,
        vocab_file: str=None,
        vocab_size: int=32000,
        min_frequency: int=2,
        special_tokens: list=["<s>", "</s>", "<pad>", "<unk>", "<mask>"],
        local_data: str=None,
        dataset_name: str="wikimedia/wikipedia",
        dataset_subset: list=['20231101.ru', '20231101.zh', '20231101.fr', '20231101.es', '20231101.en', '20231102.ar'],
        dataset_ratio: float=0.1,
        remove_columns: list=["id", "url", "title"],
        tokenizer_path: str="./"
    ):
    
        super(SparrowTokenizer, self).__init__()
        
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.local_data = local_data
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.dataset_ratio = dataset_ratio
        self.remove_columns = remove_columns
        self.tokenizer_path = Path(tokenizer_path)
        
    def process_func(self, examples):
        titles = examples["title"]
        texts = examples["text"]
        outputs = [f"{title} {text}" for title, text in zip(titles, texts)]
        return {"text": outputs}

    def create_dataset(self):

        if self.local_data is not None:
            print("[:)] Loading the data from lodal drive...")
            dataset = load_dataset("json", data_files={"data": self.local_data}, split="data[:50%]")
            chunk_size = len(dataset["text"]) // 6
            remainder = len(dataset["text"]) % 6
            self.corpus_list = [str(Path(self.tokenizer_path / f"corpus{i}.txt")) for i in range(0, 6)]
        else:
            print("[:)] Loading the data from huggingface...")
            data_list = [load_dataset(self.dataset_name, data_dir=self.dataset_subset[i], \
                split="train[:]").shuffle().train_test_split(self.dataset_ratio)["test"] for i in range(0, len(self.dataset_subset))]
            dataset = concatenate_datasets(data_list)
            dataset = dataset.map(self.process_func, batched=True, remove_columns=self.remove_columns, load_from_cache_file=False)
            chunk_size = len(dataset["text"]) // len(self.dataset_subset)
            remainder = len(dataset["text"]) % len(self.dataset_subset)
            self.corpus_list = [str(Path(self.tokenizer_path / f"corpus{i}.txt")) for i in range(0, len(self.dataset_subset))]

        print("[:)] Saving dataset...")
        start = 0
        for i in tqdm(range(0, len(self.corpus_list))):
            if os.path.isfile(self.corpus_list[i]):
                pass
            else:
                extra = 1 if i < remainder else 0
                end = start + chunk_size + extra
                with open(self.corpus_list[i], "w", encoding="utf-8") as f:
                    for example in dataset["text"][start:end]:
                        f.write(example + "\n")
        print("[:)] Dataset saved")

    
    def train(self):

        if self.vocab_file is not None:
            with open(self.vocab_file, "r", encoding="utf-8") as f:
                vocab=json.load(f)
            
            model = model.BPE(vocab=vocab, merges=[])
            tokenizer = Tokenizer(model)
        else:
            tokenizer = Tokenizer(models.BPE())

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()
    
        trainer = trainers.BpeTrainer(

            
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True
        )

        tokenizer.train(self.corpus_list, trainer)

        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            special_tokens=[("<s>", 0), ("</s>", 1)]
        )

        self.tokenizer = tokenizer

    def config(self):
        self.config = {
            "add_bos_token": False,
            "add_eos_token": False,
            "add_prefix_space": True,
            "added_tokens_decoder": {
                "0": {
                    "content": "<s>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "1": {
                    "content": "</s>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "2": {
                    "content": "<pad>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "3": {
                    "content": "<unk>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "4": {
                    "content": "<mask>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                }
            },
            "additional_special_tokens": [],
            "bos_token": "<s>",
            "clean_up_tokenization_spaces": False,
            "eos_token": "</s>",
            "legacy": True,
            "model_max_length": 1000000000000000019884624838656,
            "pad_token": None,
            "sp_model_kwargs": {},
            "spaces_between_special_tokens": False,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "<unk>",
            "use_default_system_prompt": False,
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
        }
    
    def save(self):
        if not hasattr(self, "tokenizer"):
            raise ValueError("[:(] Train tokenizer first before saving!")
        else:
            pass
        
        self.config()
        self.tokenizer.save(str(Path(self.tokenizer_path / "tokenizer.json")))
        self.tokenizer.model.save(str(self.tokenizer_path))
        with open(str(Path(self.tokenizer_path / "tokenizer_config.json")), "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
        
        print("[:)] All files are saved")
    
    def evaluate(self):
        tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
        message = "你好，世界 + Hello World + Bonjour + Hola + Привет + مرحبًا"
        message_enc = tokenizer.encode(message)
        message_dec = tokenizer.decode(message_enc)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="Continual Pretraining Argument YAML Location")
    args_parser.add_argument("--args", type=str, required=True, default="./params.yaml")
    args_parser = args_parser.parse_args()
    args = yaml.safe_load(Path(f"{args_parser.args}").read_text())
    tokenizer = SparrowTokenizer(**args)
    tokenizer.create_dataset()
    tokenizer.train()
    tokenizer.save()