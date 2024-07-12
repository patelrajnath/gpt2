import inspect
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.onnx.utils import model_signature
from transformers import GPT2LMHeadModel
import tiktoken
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from hellaswag import iterate_examples, render_example


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.GPT_INIT_SCALE = 1
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size).
        #                                         view(1, 1, config.block_size, config.block_size)))

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
        if master_process:
            print(f'num params decayed: {len(decay_params)} with {num_decay_params} parameters')
            print(f'num params nodecayed: {len(nodecay_params)} with {num_nodecay_params} parameters')
        optim_groups = [
            {'params': decay_params, 'weight)decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
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
        if master_process:
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


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt


class DataLoaderEDU10B(object):
    def __init__(self, B, T, process_rank, num_processes, split):
        self.current_position = None
        self.tokens = None
        self.current_shard = None
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # Get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split:{split}"

        if master_process:
            print(f'found {len(shards)} shards for split {split}')
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += self.current_position + (B * T * self.num_processes + 1)
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.current_position = self.B * self.T * self.process_rank
            self.tokens = load_tokens(self.shards[self.current_shard])
        return x, y


# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


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

device_type = 'cuda' if device.startswith('cuda') else 'cpu'

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding('gpt2')

max_lr = 6e-4
min_lr = max_lr * 0.1
warmpu_steps = 715
max_steps = 19073
B = 16
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

use_compile = False
if torch.cuda.is_available() and use_compile:
    model_gpt = torch.compile(model_gpt)

raw_model = model_gpt.module if ddp else model_gpt

train_loader = DataLoaderEDU10B(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderEDU10B(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=max_lr, device=device)
# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:  # open for writing to clear the file
    pass


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
    last_step = (step == max_steps - 1)
    # Calculate validation loss after 100 steps
    if step % 250 == 0 or last_step:
        model_gpt.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                logits, loss = model_gpt(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss

            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f'validation loss:{val_loss_accum.item:.4f}')
    if step > 0 and step % 250 == 0 or last_step:
        model_gpt.eval()
        num_return_sequences = 5
        max_length = 30
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        x = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = x.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)

        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits = model_gpt(x)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_prob, topk_idices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_prob, 1, generator=sample_rng)
                xcol = torch.gather(topk_idices, -1, ix)
                x = torch.cat((x, xcol), dim=1)

        for i in range(num_return_sequences):
            tokens = x[i, : max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank:{ddp_rank} sample:", decoded)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model_gpt(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_steps in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
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
            f'step:{step:4d} | loss: {loss_accum.item():.6f} | lr :{lr:.6f} | time: {(t2 - t1) * 1000:.4f}ms | norm: {norm:0.4f} | '
            f'token/sec:{(train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t2 - t1):.4f}')

# model_gpt = GPT.from_pretrained('gpt2')
# model_gpt = GPT2LMHeadModel.from_pretrained('gpt2')
# print('model loaded')
# model_gpt.eval()
# model_gpt.to('cuda')
# import tiktoken
#
