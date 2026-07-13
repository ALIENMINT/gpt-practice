from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_emb % config.n_head == 0
        self.c_attn=nn.Linear(config.n_emb,3*config.n_emb)
        self.c_proj=nn.Linear(config.n_emb,config.n_emb)
        self.n_head=config.n_head
        self.n_emb =config.n_emb
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size,config.block_size))
                .view(1,1,config.block_size,config.block_size) # (1,1,,) because pytorch can boardcast it to (batch_size,n_head) 
        )

    def forward(self,x):
        B,T,C=x.size() 
        #in GPT-2 (124M), n_head=12, head_size=64, n_emb==nh*hs=768
        qkv=self.c_attn(x) # x @ W.T + b
        q,k,v=qkv.split(self.n_emb,dim=2) # (B,T,3C)=>3(B,T,C)
        # multihead
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B,T,768)=>(B,12,T,64)==(B,nd,T,hs)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        # attention
        att=((q @ k.transpose(-2,-1)) #(B,12,T,64)*(B,12,64,T)=>(B,12,T,T)
             * (1.0 / math.sqrt(k.size(-1)))) # scale down, prevent var from overlarge, *(1.0/sqrt(64))
        att=att.masked_fill(self.bias[:,:,T,T]==0,float('-inf'))
        att=F.softmax(att,dim=-1)
        y=att @ v                                  #(B,12,T,T)*(B,12,T,64)=>(B,12,T,64)
        y=y.transpose(1,2).contiguous().view(B,T,C)#(B,12,T,64)=>(B,T,12,64)=>(B,T,C)
        y=self.c_proj(y)                           #(B,T,C)=>(B,T,C), to mix the independent k,q,v
        return y
        # query: what info in other pos should I pay attention to in this pos?
        # key: index, tell other pos what key words they can find in this pos.
        # value: content, real info

# params = 768*(3*768)

class MLP(nn.Module): # FFN
    def __init__(self,config):
        super().__init__()
        self.c_fc  =nn.Linear(config.n_emb,4* config.n_emb) #fully connected
        self.gelu  =nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4* config.n_emb,config.n_emb)
    def forward(self,x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_emb) # weights, bias, 2*n_emb
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_emb)
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
    n_emb:int=768

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_emb),
            wpe=nn.Embedding(config.block_size,config.n_emb),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),#12
            ln_f=nn.LayerNorm(config.n_emb),
        ))
        self.lm_head=nn.Linear(config.n_emb,config.vocab_size,bias=False)

    def forward(self,idx):
        #idx.shape==(B,T)
        B,T=idx.size()
        assert T<=self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}."
        
        # forward token and pos emb
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device) # shape (T,)
        pos_emb=self.transformer.wpe(pos) # (T,)  =>   (T,n_emb)
        tok_emb=self.transformer.wte(idx) # (B,T) => (B,T,n_emb)
        x=pos_emb+tok_emb #(T,n_emb)+(B,T,n_emb) => broadcast (B,T,n_emb)+(B,T,n_emb)
        
        # forward blocks of the transformer
        for block in self.transformer.h:
            x=block(x)
        
        #forward ln_f and the classfier
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)
        return logits
 
    @classmethod
    def from_pretrained(cls,model_type):
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpe2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args={
            'gpt2':        dict(n_layer=12,n_head=12,n_emb=768),#124M
            'gpt2-medium': dict(n_layer=24,n_head=16,n_emb=1024),#350M
            'gpt2-large':  dict(n_layer=36,n_head=20,n_emb=1280),#774M
            'gpt2-xl':     dict(n_layer=48,n_head=25,n_emb=1600),#1550M
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

# ---------------------------
num_return_sequence=5
max_length=30

model=GPT.from_pretrained('gpt2')
model.eval()
model.to('cuda')

# prefix tokens
import tiktoken
enc=tiktoken.get_encoding('gpt2')
tokens=enc.encode("Hello, I'm a language model,")
tokens=torch.tensor(tokens,dtype=torch.long) #(8,)
tokens=tokens.unsqueeze(0).repeat(num_return_sequence,1) #(5,8)
x=tokens.to('cuda')

torch.manual_seed(37)
torch.cuda.manual_seed(37)
while x.size(1)<max_length:
    logits