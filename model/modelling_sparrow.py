import math
from typing import Optional

import torch 
import torch.nn as nn
import torch.nn.functional as F 

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

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

## RoPE - from https://arxiv.org/pdf/2104.09864v5
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    
    cos = cos.unsqueeze(unsqueeze_dim) 
    sin = sin.unsqueeze(unsqueeze_dim)
   
    q_embed = (q*cos) + (rotate_half(q)*sin) 
    k_embed = (k*cos) + (rotate_half(k)*sin) 
    
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super(RotaryEmbedding, self).__init__()
        self.hidden_size = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))  
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t @ inv_freq.unsqueeze(0) 
        freqs = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
        
    def forward(self, q, k):
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0) 
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)
    

## RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1.0e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def normalize(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self.normalize(x).type_as(x)
        return output * self.weight

def repeat_kv(x, n_rep):
    batch, length, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    
    x = x[:, :, :, None, :].expand(batch, length, num_key_value_heads, n_rep, head_dim)
    return x.reshape(batch, length, num_key_value_heads * n_rep, head_dim)

## SparrowAttention
class SparrowAttention(nn.Module):
    '''
    '''
    def __init__(self, config: SparrowConfig=None):
        super(SparrowAttention, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.vocab_size = config.vocab_size
        self.dropout = config.dropout
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
    
        self.wq = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=self.config.attention_bias)
        self.wk = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.config.attention_bias)
        self.wv = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.config.attention_bias)
        self.wo = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=self.config.attention_bias)
        self.k_cache, self.v_cache = None, None
        self.attention_dropout = nn.Dropout(self.dropout)
        self.residual_dropout = nn.Dropout(self.dropout)
    
    def forward(self, x: torch.Tensor, use_kv_cache=False):
        b, s = x.shape[:2]
        if use_kv_cache and self.eval():
            if self.k_cache is None or self.k_cache.shape[1] != s-1:
                q, k, v = self.wq(x), self.wk(x), self.wv(x)
            else:
                token = x[:, -1:, :]
                q = torch.cat((torch.zeros_like(x[:, :-1, :]), self.wq(token)), dim=1) 
                k = torch.cat((self.k_cache, self.wk(token)), dim=1)
                v = torch.cat((self.v_cache, self.wv(token)), dim=1)
            
            self.k_cache, self.v_cache = k, v
        else:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = q.view(b, s, self.num_attention_heads, self.head_dim)
        k = k.view(b, s, self.num_key_value_heads, self.head_dim)
        v = v.view(b, s, self.num_key_value_heads, self.head_dim)
        q, k = self.rotary_emb(q, k)
        k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.config.flash_attn:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                    dropout_p=self.dropout if self.training else 0.0, 
                                                    is_causal=True)
        else:
            mask = torch.full((1, 1, self.config.max_seq_len, self.config.max_seq_len), float("-inf"), device=x.device) 
            mask = torch.triu(mask, diagonal=1)
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores + mask[:, :, :s, :s]
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            scores = self.attention_dropout(scores)
            output = torch.matmul(scores, v) 
        
        output = output.transpose(1, 2).contiguous().view(b, s, -1)
        output = self.wo(output)
        output = self.residual_dropout(output)
        return output

class SparrowLinear(nn.Module):
    def __init__(self, config: SparrowConfig=None):
        super(SparrowLinear, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_dim = config.intermediate_dim
        self.gate = nn.Linear(self.hidden_size, self.intermediate_dim, bias=self.config.mlp_bias)
        self.up = nn.Linear(self.hidden_size, self.intermediate_dim, bias=self.config.mlp_bias)
        self.out = nn.Linear(self.intermediate_dim, self.hidden_size, bias=self.config.mlp_bias)
    
    def forward(self, x):
        return self.out(F.silu(self.gate(x)) * self.up(x))

class SparrowDecoderLayer(nn.Module):
    def __init__(self, config: SparrowConfig=None, layer_idx: int=None):
        super(SparrowDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention = SparrowAttention(config=config)
        self.linear = SparrowLinear(config=config)
        self.input_norm = RMSNorm(dim=config.hidden_size)
        self.pos_attn_norm = RMSNorm(dim=config.hidden_size)
        self.layer_idx = layer_idx
    
    def forward(self, x, use_kv_cache):
        residual = x
        x = self.input_norm(x)
        residual, x = x, self.attention(x=x, use_kv_cache=use_kv_cache) + residual
        x = self.linear(self.pos_attn_norm(x))
        x = x + residual
        return x

class SparrowModel(PreTrainedModel):
    config_class = SparrowConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.num_hidden_layers = self.config.num_hidden_layers
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)

        self.decoder = nn.ModuleList()
        for layer_idx in range(self.num_hidden_layers):
            self.decoder.append(SparrowDecoderLayer(config=self.config, layer_idx=layer_idx))
        
        self.norm = RMSNorm(dim=self.config.hidden_size)
        self.apply(self.weights_init)
    
    def weights_init(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(self, input_ids, use_kv_cache=False):
        x = self.dropout(self.token_embedding(input_ids))
        
        for idx, layer in enumerate(self.decoder):
            x = layer(x=x, use_kv_cache=use_kv_cache)
        
        return self.norm(x)

class SparrowModelForCausalLM(SparrowModel):
    def __init__(self, config):
        super().__init__(config)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=self.config.mlp_bias)
        self.token_embedding.weight = self.output.weight
        self.loss = None

        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.num_hidden_layers))
        
    def forward(self, input_ids, labels=None, use_kv_cache=False):
        x = super().forward(input_ids, use_kv_cache)
        
        if labels is not None:
            logits = self.output(x)  
            self.loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0) 
        else:
            logits = self.output(x[:, [-1], :])  
            self.loss = None  

        return CausalLMOutputWithPast(self.loss, logits)
    
    @torch.no_grad()
    def generate(self, input_ids, eos=1, max_new_tokens=50, temperature=0.7, top_k=None, repetition_penalty=1.,
                 use_kv_cache=True, use_beam_search=False, beam_size=3):
        s = input_ids.shape[1]
        
        if use_beam_search:
            sequences = [(input_ids, 0)]  # List of (sequence, cumulative log probability)
            for _ in range(max_new_tokens - 1):
                all_candidates = []
                for seq, score in sequences:
                    inference_res = self(seq, labels=None, use_kv_cache=use_kv_cache)
                    logits = inference_res.logits[:, -1, :]
                    
                    if repetition_penalty != 1.0:
                        for token in set(seq.tolist()[0]):
                            logits[:, token] /= repetition_penalty
                    
                    logits = logits / temperature if temperature > 0 else logits
                    probs = F.log_softmax(logits, dim=-1)
                    top_log_prob, idx_next = torch.topk(probs, beam_size, dim=-1)
                    
                    for i in range(beam_size):
                        next_seq = torch.cat((seq, idx_next[:, i].unsqueeze(1)), dim=1)
                        next_score = score + top_log_prob[:, i].item()
                        all_candidates.append((next_seq, next_score))
                
                sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
                if all(seq[0][:, -1].item() == eos for seq in sequences):
                    break
            
            best_seq = sequences[0][0]
            return best_seq.tolist()[0][s:]
        
        # Greedy search (default)
        generated_tokens = []
        while len(generated_tokens) < max_new_tokens - 1:
            inference_res = self(input_ids, labels=None, use_kv_cache=use_kv_cache)
            logits = inference_res.logits[:, -1, :]
            
            if repetition_penalty != 1.0:
                for token in set(input_ids.tolist()[0]):
                    logits[:, token] /= repetition_penalty
            
            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            if idx_next.item() == eos:
                break
            
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            generated_tokens.append(idx_next.item())
        
        return generated_tokens