from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1 # type: ignore
        self.n_head=config.n_head
        self.n_embd =config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size,config.block_size))
                .view(1,1,config.block_size,config.block_size) # (1,1,,) because pytorch can boardcast it to (batch_size,n_head) 
        )

    def forward(self,x):
        B,T,C=x.size() 
        #in GPT-2 (124M), n_head=12, head_size=64, n_embd==nh*hs=768
        qkv=self.c_attn(x) # x @ W.T + b
        q,k,v=qkv.split(self.n_embd,dim=2) # (B,T,3C)=>3(B,T,C)
        # multihead
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B,T,768)=>(B,12,T,64)==(B,nd,T,hs)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        # attention
        att=((q @ k.transpose(-2,-1)) #(B,12,T,64)*(B,12,64,T)=>(B,12,T,T)
             * (1.0 / math.sqrt(k.size(-1)))) # scale down, prevent var from overlarge, *(1.0/sqrt(64))
        att=att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        att=F.softmax(att,dim=-1)
        y=att @ v                                  #(B,12,T,T)*(B,12,T,64)=>(B,12,T,64)
        y=y.transpose(1,2).contiguous().view(B,T,C)#(B,12,T,64)=>(B,T,12,64)=>(B,T,C)
        y=self.c_proj(y)                           #(B,T,C)=>(B,T,C), mix k,q,v to get single head attn
        return y
        # query: what info in other pos should I pay attention to in this pos?
        # key: index, tell other pos what key words they can find in this pos.
        # value: content, real info

# params = 768*(3*768)

class MLP(nn.Module): # FFN
    def __init__(self,config):
        super().__init__()
        self.c_fc  =nn.Linear(config.n_embd,4* config.n_embd) #fully connected
        self.gelu  =nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4* config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1 # type: ignore
    def forward(self,x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd) # weights, bias, 2*n_embd
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)
    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x
    
@dataclass
class GPTConfig:
    block_size: int =1024
    vocab_size:int=50257# 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer:int=12
    n_head:int=12
    n_embd:int=768

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),#12
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)

        # weight sharing scheme
        # 768*50257=28,597,376, about 30% of 124M, huge usage decline!
        self.transformer.wte.weight=self.lm_head.weight # data_ptr() points the same address

        # init params
        self.apply(self._init_weight)

    def _init_weight(self,module):
        if isinstance(module,nn.Linear):
            std=0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                std*=(2*self.config.n_layer)**-0.5 # n**-0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None):
        #idx.shape==(B,T)
        B,T=idx.size()
        assert T<=self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}."
        
        # forward token and pos emb
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device) # shape (T,)
        pos_emb=self.transformer.wpe(pos) # (T,)  =>   (T,n_embd)
        tok_emb=self.transformer.wte(idx) # (B,T) => (B,T,n_embd)
        x=pos_emb+tok_emb #(T,n_embd)+(B,T,n_embd) => broadcast (B,T,n_embd)+(B,T,n_embd)
        
        # forward blocks of the transformer
        for block in self.transformer.h:
            x=block(x)
        
        #forward ln_f and the classfier
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1)) # softmax + -log p
        return logits,loss
 
    @classmethod
    def from_pretrained(cls,model_type):
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpe2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args={
            'gpt2':        dict(n_layer=12,n_head=12,n_embd=768),#124M
            'gpt2-medium': dict(n_layer=24,n_head=16,n_embd=1024),#350M
            'gpt2-large':  dict(n_layer=36,n_head=20,n_embd=1280),#774M
            'gpt2-xl':     dict(n_layer=48,n_head=25,n_embd=1600),#1550M
        }[model_type]
        config_args['vocab_size']=50257
        config_args['block_size']=1024

        config=GPTConfig(**config_args)
        model=cls(config)
        sd=model.state_dict()
        sd_keys=sd.keys()
        sd_keys=[k for k in sd_keys if not k.endswith('.attn.bias')] #filter out register_buffer(causal mask)

        #huggingface model
        model_hf=GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf=model_hf.state_dict()

        #
        sd_keys_hf=sd_hf.keys()
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed=['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']

        # # print keys
        # print("=== HF keys ===")
        # print(sorted(sd_keys_hf))
        # print("=== My model keys ===")
        # print(sorted(sd_keys))
        # # what keys I lost?
        # hf_set = set(sd_keys_hf)
        # my_set = set(sd_keys)
        # print("In HF but not in mine:", hf_set - my_set)
        # print("In mine but not in HF:", my_set - hf_set)

        assert len(sd_keys_hf)==len(sd_keys), f"mismatched keys:{len(sd_keys_hf)}!={len(sd_keys)}"
        for k in sd_keys_hf:
            # do transpose because these params are stored as (out,in) in hf, but we need (in,out)
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1]==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

# ---------------------------------------
# prefix tokens
import tiktoken
class DataLoderLite:
    def __init__(self,B,T):
        self.B=B
        self.T=T

        with open('/home/soren/project/GPT/dataset/input.txt','r') as f:
            text=f.read()
        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches") # how many batches are needed to train one epoch

        self.current_position=0
    
    def next_batch(self):
        B,T=self.B,self.T
        buf=self.tokens[self.current_position:self.current_position + B*T + 1]
        x=buf[:-1].view(B,T)
        y=buf[1:].view(B,T)
        self.current_position += B*T
        # if out of bounds, reset
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position=0
        return x,y
    
# ---------------------------
num_return_sequence=5
max_length=30
torch.manual_seed(37)
torch.cuda.manual_seed(37)

# device select
device='cpu'
if torch.cuda.is_available():
    device='cuda'
elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
    device='mps'
print(f"using device: {device}")

# get tokens
train_loader=DataLoderLite(B=1,T=1024)

# get logits
model=GPT(GPTConfig())
model.to(device)

import time
# optimize grad
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
for i in range(50):
    t0=time.time()
    x,y=train_loader.next_batch()
    x,y=x.to(device),y.to(device)
    optimizer.zero_grad()
    logits,loss=model(x,y)
    loss.backward() # calculate grad
    optimizer.step() # update grad
    torch.cuda.synchronize()
    t1=time.time()
    dt=(t1-t0)*1000
    print(f"batch {i}, loss: {loss.item()}, dt: {dt}ms")


import sys; sys.exit(0)