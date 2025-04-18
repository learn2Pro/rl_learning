import torch
import torch.nn.functional as F
import torch.nn as nn

from dataclasses import dataclass
from transformers import PretrainedConfig, AutoConfig, AutoModelForCausalLM
from transformers import PreTrainedModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
FLASH = 0

# rope
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = theta ** -(torch.arange(0, dim, 2)[:dim//2].float() / dim)
    t = torch.arange(end)
    freqs = torch.outer(t, freqs) # m * \theta
    # freqs = t * freqs
    freqs = torch.polar(torch.ones_like(freqs), freqs) # cos(m * \theta) + jsin(m * \theta)
    return freqs

# 2. 为广播 reshape freqs
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    if freqs_cis.shape[0] > x.shape[1]:
        freqs_cis = freqs_cis[:x.shape[1]]
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [1 if i != 1 and i != x.ndim - 1 else x.shape[i] for i in range(x.ndim)]
    return freqs_cis.view(*shape).to(x.device)

def apply_rotary_emb(q, k, freqs):
    xq = torch.view_as_complex(q.view(*q.shape[:-1], -1, 2)) # batch, seq_len, n_head, dim//2
    xk = torch.view_as_complex(k.view(*k.shape[:-1], -1, 2)) # batch, seq_len, n_head, dim//2
    
    freqs_cis = reshape_for_broadcast(freqs, xq) # freqs_cis.shape = (1,x.shape[1],1,x.shape[-1])

    xq_out = torch.view_as_real(xq * freqs_cis).flatten(3) # batch, seq_len, n_head, dim
    xk_out = torch.view_as_real(xk * freqs_cis).flatten(3) # batch, seq_len, n_head, dim

    return xq_out.type_as(q), xk_out.type_as(k)
    
@dataclass
class GPTConfig(PretrainedConfig):
    n_block: int
    n_layer: int
    n_head: int
    n_embed: int
    n_vocab: int = 50257
    model_type: str = 'buddygpt'
    pad_token_id=None,
    bos_token_id=None,
    eos_token_id=50256,
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)



class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)  # split last dim into two
        return F.silu(x1) * x2  # silu == swish
        
class SelfCausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.n_block = config.n_block
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.freqs_cis = precompute_freqs_cis(config.n_embed // config.n_head, config.n_block * 2)
        self.register_buffer('tril', torch.tril(torch.ones(config.n_block, config.n_block)).view(1,1,config.n_block,config.n_block))

    def forward(self, x):
        B, T, _ = x.size()
        attn = self.c_attn(x)
        q, k, v = attn.split(self.n_embed, dim=-1) # B,n_block,n_embed
        q = q.view(B, T, self.n_head, -1).transpose(1, 2) # B, n_head, n_block, n_embed//n_head
        k = k.view(B, T, self.n_head, -1).transpose(1, 2) # B, n_head, n_block, n_embed//n_head
        v = v.view(B, T, self.n_head, -1).transpose(1, 2) # B, n_head, n_block, n_embed//n_head
        q, k = apply_rotary_emb(q, k, self.freqs_cis)
        if FLASH:
            o_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            qk = q @ k.transpose(-2, -1)
            qk = qk.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf'))
            o_attn = (F.softmax(qk, dim=-1) * (self.n_embed ** -0.5)) @ v
        o_attn = o_attn.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(o_attn)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embed
        self.ln1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU()
        self.ln2 = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.ln2(x)
        return x

class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = SelfCausalAttention(config)
        self.mlp = MLP(config)
        self.norm1 = nn.RMSNorm(config.n_embed)
        self.norm2 = nn.RMSNorm(config.n_embed)

    def forward(self, x):
        x = x + self.mha(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class BuddyGPT(PreTrainedModel):
    config_class = GPTConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.n_vocab = config.n_vocab
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.n_vocab, config.n_embed),
            wpe = nn.Embedding(config.n_block, config.n_embed),
            layers = nn.ModuleList([Layer(config) for _ in range(config.n_layer)]),
            ln_norm = nn.RMSNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)
        self.eos_token_id = config.eos_token_id
        self.transformer.wte.weight = self.lm_head.weight
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 / (2*self.config.n_layer) ** 0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)
            

    def forward(self, input_ids, labels=None, **kwargs):
        input_ids = input_ids.to(device)
        B, T = input_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        token_embed = self.transformer.wte(input_ids)
        pos_embed = self.transformer.wpe(pos)
        x = token_embed + pos_embed
        for layer in self.transformer.layers:
            x = layer(x)
        x = self.transformer.ln_norm(x)

        if labels is not None:
            labels = labels.to(device)
            logits = self.lm_head(x)
            shape_logits = logits[:,:-1,:].contiguous().view(-1, self.n_vocab)
            targets = labels[:,1:].contiguous().view(-1)
            loss = F.cross_entropy(shape_logits, targets, ignore_index=-100)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return (loss, logits) if loss else logits

    @torch.no_grad()
    def generate(self, input_ids, max_length, temperature=1.0, **kwargs):
        x = input_ids
        for _ in range(max_length):
            idx_cond = x if x.size(1)<=self.config.n_block else x[:, -self.config.n_block:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature # last token
            probs = F.softmax(logits, dim=-1) # B, n_vocab
            predict = torch.multinomial(probs, num_samples=1) # B, 1
            if self.eos_token_id and self.eos_token_id == predict.item():
                return x
            x = torch.cat([x, predict], dim=-1)
        return x

AutoConfig.register("buddygpt", GPTConfig)
AutoModelForCausalLM.register(GPTConfig, BuddyGPT)

