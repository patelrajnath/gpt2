import inspect
import math
import os
import time
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import tiktoken
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.GPT_INIT_SCALE = 1
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size).
                                                view(1, 1, config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
        # att = att.masked_fill(self.bias[:, :, :T, : T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.GPT_INIT_SCALE = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embed),
            wpe=nn.Embedding(config.block_size, config.n_embed),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embed),
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # NOTE:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT_INIT_SCALE'):
                std = (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        B, T = idx.size()

        assert T <= self.config.block_size, f'Cannot forwad a sequence of length:{T}, ' \
                                            f'max allowed: {self.config.block_size}'
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = 0
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        print(f'Loading weights from pretrained gpt:{model_type}')
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embed=1600)
        }[model_type]

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()

        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()

        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight', ]
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizer(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f'num params decayed: {len(decay_params)} with {num_decay_params} parameters')
        print(f'num params nodecayed: {len(nodecay_params)} with {num_nodecay_params} parameters')
        optim_groups = [
            {'params': decay_params, 'weight)decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f'using fused AdamW:{use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer


class DataLoaderLite(object):
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        with open('input.txt', 'r', encoding='utf8') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)}')

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += self.current_position + (B * T * self.num_processes + 1)
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

from torch.distributed import init_process_group, destroy_process_group

# Setup DDP
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
# torchrun command sets the env variable RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    # use of ddp requires CUDA, and we set the device appropriately according to the RANK
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device:{device}')

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

num_return_sequences = 5
max_length = 30
max_lr = 6e-4
min_lr = max_lr * 0.1
warmpu_steps = 10
max_steps = 50
B = 4
T = 1024
total_batch_size = 524288  # 2**19
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f'total desired batch size:{total_batch_size}')
    print(f'total gradient accumulation steps: {grad_accum_steps}')

model_gpt = GPT(GPTConfig(vocab_size=50304))
model_gpt.to(device)

if ddp:
    model_gpt = DDP(model_gpt, device_ids=[ddp_local_rank])

if torch.cuda.is_available():
    model_gpt = torch.compile(model_gpt)

raw_model = model_gpt.module if ddp else model_gpt

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=max_lr, device=device)


def get_lr(it):
    if it < warmpu_steps:
        return max_lr * (it + 1) / warmpu_steps
    if it > max_steps:
        return min_lr
    decay_rate = (it - warmpu_steps) / (max_steps - warmpu_steps)
    assert 0 <= decay_rate <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_rate))
    return min_lr + coeff * (max_lr - min_lr)


for step in range(max_steps):
    t1 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_steps in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model_gpt(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model_gpt.require_backward_grad_sync = (micro_steps == grad_accum_steps - 1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model_gpt.parameters(), 1.0)

    lr = get_lr(step)
    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr

    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.time()
    if master_process:
        print(
            f'step:{step} | loss: {loss_accum.item():.6f} | lr :{lr:.6f} | time: {(t2 - t1) * 1000}ms | norm: {norm:0.4f} | '
            f'token/sec:{(train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t2 - t1)}')

# model_gpt = GPT.from_pretrained('gpt2')
# model_gpt = GPT2LMHeadModel.from_pretrained('gpt2')
# print('model loaded')
# model_gpt.eval()
# model_gpt.to('cuda')
# import tiktoken
#
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
#
# tokens = torch.tensor(tokens, dtype=torch.long)
# x = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to('cuda')
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
#
# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits = model_gpt(x)
#         logits = logits[:, -1, :]
#         probs = F.softmax(logits, dim=-1)
#         topk_prob, topk_idices = torch.topk(probs, 50, dim=-1)
#         ix = torch.multinomial(topk_prob, 1)
#         xcol = torch.gather(topk_idices, -1, ix)
#         x = torch.cat((x, xcol), dim=1)
#
# for i in range(num_return_sequences):
#     tokens = x[i, : max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)
