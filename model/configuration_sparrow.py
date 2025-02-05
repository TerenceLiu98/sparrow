from typing import Optional
from transformers import PretrainedConfig

class SparrowConfig(PretrainedConfig):
    model_type = "sparrow"

    def __init__(
        self,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 16,
        num_key_value_heads: Optional[int] = None,
        max_seq_len: int = 512,
        attention_bias: bool = False,
        flash_attn: bool = True,
        vocab_size: int = 32000,
        hidden_dim: Optional[int] = None,
        intermediate_dim: int = 2048, 
        norm_eps: float = 1e-5,
        mlp_bias: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # attention args
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.max_seq_len = max_seq_len
        self.attention_bias = attention_bias
        self.flash_attn = flash_attn
        # mlp args 
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim if hidden_dim is not None else hidden_size
        self.intermediate_dim = intermediate_dim
        self.norm_eps = norm_eps
        self.mlp_bias = mlp_bias
        self.dropout = dropout
