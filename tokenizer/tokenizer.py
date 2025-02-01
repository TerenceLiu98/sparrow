from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from tokenizers import decoders, models, pre_tokenizers, processors, trainers, Tokenizer



class SparrowTokenizer:
    def __init__(
        self,
        vocab_file: str=None,
        vocab_size: int=256000,
        min_frequency: int=2,
        special_tokens: list=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        dataset_name: str="wikimedia/wikipedia",
        dataset_subset: list=['20231101.ru', '20231101.zh', '20231101.fr', '20231101.es', '20231101.en', '20231102.ar'],
        remove_columns: list=["id", "url", "title"],
        tokenizer_path: str="./"
    ):
    
        super(SparrowTokenizer, self).__init__()
        
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.remove_columns = remove_columns
        self.tokenizer_path = Path(tokenizer_path)
        
    def process_func(self, examples):
        titles = examples["title"]
        texts = examples["text"]
        outputs = [f"{title} {text}" for title, text in zip(titles, texts)]
        return {"text": outputs}

    def create_dataset(self, ratio=0.1):
        data_list = [load_dataset(self.dataset_name, data_dir=self.dataset_subset[i], \
            split="train[:{}%]".format(int(ratio * 100))) for i in range(0, len(self.dataset_subset))]
        dataset = concatenate_datasets(data_list)
        dataset = dataset.map(self.process_func, batched=True, remove_columns=self.remove_columns)

        print("[:)] Saving dataset...")
        with open(Path(self.tokenizer_path / "corpus.txt"), "w", encoding="utf-8") as f:
            for example in dataset["text"]:
                f.write(example + "\n")
        print("[:)] Dataset saved")

    
    def train(self):

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

        tokenizer.train([str(Path(self.tokenizer_path / "corpus.txt"))], trainer)

        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            special_tokens=[("<s>", 0), ("</s>", 1)]
        )

        self.tokenizer = tokenizer
    
    def save(self):
        if not hasattr(self, "tokenizer"):
            raise ValueError("[:(] Train tokenizer first before saving!")
        else:
            pass

        self.tokenizer.model.save(".", prefix="tokenizer")
        os.rename("tokenizer-vocab.json", str(Path(self.tokenizer_path / "tokenizer-vocab.json")))
        os.rename("tokenizer-merges.txt", str(Path(self.tokenizer_path / "tokenizer-merges.txt")))