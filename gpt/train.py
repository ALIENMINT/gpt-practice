import torch
import torch.nn as nn

from torch.nn import functional as F

device='cuda' if torch.cuda.is_available() else 'cpu'
batch_size=64
block_size=256
max_iters=4000
eval_interval=500
eval_batches=200
learning_rate=3e-4
n_emb=384 #n_heads * 64
n_heads=6
n_layer=6
dropout=0.2
torch.manual_seed(42)

with open('input.txt','r',encoding='utf-8') as f:
    text=f.read()

# vocab dictionary
chars=sorted(list(set(text)))
vocab_size=len(chars)
stoi={ch:i for i,ch in enumerate(chars)}#exp a:1 b:2
itos={i:ch for i,ch in enumerate(chars)}#exp 1:a 2:b
encoder=lambda s: [stoi[c] for c in s]# string=>list[int]
decoder=lambda l: "".join(itos[i] for i in l)# list=>string

# data splits
data=torch.tensor(encoder(text),dtype=torch.long)
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

# data loading 
def get_batch(split):
    data=train_data if split == "train" else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,)) # (batch_size,) or batch_size #[0:len-block_size)
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x=x.to(device)
    y=y.to(device)
    return x,y

@torch.no_grad()
def eval_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_batches) #storage loss of evert step
        for k in range(eval_batches):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item() #tensor => float
        out[split]=losses.mean()
    return out

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_emb,head_size,bias=False)
        self.query=nn.Linear(n_emb,head_size,bias=False)
        self.value=nn.Linear(n_emb,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))# 1.efficiency, this is a tempplate 2.compatibilty, ensure tril mask is in the gpu 
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        B,T,C=x.shape
        q=self.query(x) #(B,T,C)=>(B,T,head_size)
        k=self.key(x) #(B,T,C)=>(B,T,head_size)
        v=self.value(x) #(B,T,C)=>(B,T,head_size)
        #attention scores "affinities" Q*K.T
        wei=q@k.transpose(-2,-1) *C**-0.5 #(B,T,head_size)*(B,head_size,T)=>(B,T,T) # *C**-0.5 to ensure the sigma == 1
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1) #obtain the probability distribution
        wei=self.dropout(wei)
        # perform the weighted aggragation of the values
        out=wei@v#(B,T,T)@(B,T,head_size)=>(B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj=nn.Linear(n_emb,n_emb) # mix the features
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self,n_emb):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_emb,n_emb*4),
            nn.ReLU(),# ReLU: x<0,f=0;x>0,f=x
            nn.Linear(n_emb*4,n_emb)
            )
    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self,n_emb,n_heads):
        super().__init__()
        self.sa=MultiHeadAttention(n_heads,n_emb//n_heads)
        self.ffwd=FeedFoward(n_emb)
        self.ln1=nn.LayerNorm(n_emb)
        self.ln2=nn.LayerNorm(n_emb)
    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x)) # pre-LN, more stable than post-LN in the attention paper
        return x

class BigramLanguageModel(nn.Module):
    # build vocab vector table
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_emb)
        self.position_embedding_table=nn.Embedding(block_size,n_emb)
        self.blocks=nn.Sequential(*[Block(n_emb,n_heads) for _ in range(n_layer)]) # * is unpacking operator, so Sequential() expects nn.Module rather than a list
        self.ln_f=nn.LayerNorm(n_emb)
        # self.sa_head=MultiHeadAttention(n_heads,n_emb//n_heads)#exp 4 heads of 8 dim self-attention
        # self.ffwd=FeedFoward(n_emb)# linear => unlinear
        self.lm_head=nn.Linear(n_emb,vocab_size)

    # find the most possible next char from token_embedding_table, eval the loss
    def forward(self,idx,targets=None): # idx [B,T], targets [B,T]
        B,T=idx.shape

        #logits
        tok_emb=self.token_embedding_table(idx)# (B,T,C)
        pos_emb=self.position_embedding_table(torch.arange(T,device=device))#(T,C)
        x=tok_emb+pos_emb#(B,T,C)
        #x=self.sa_head(x)
        #x=self.ffwd(x)#(B,T,C)
        x=self.blocks(x) #(B,T,C)
        x=self.ln_f(x)
        logits=self.lm_head(x)# logits [B,T,vocab_size]

        #loss
        if targets==None:
            loss=None
        else:
            B,T,C=logits.shape
            #cross_entropy expects logits of shape (N,C) and targets of shape (N,)
            logits=logits.view(B*T,C) 
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    
    # generate the new chars
    def generate(self,idx,max_new_tokens):# idx [B,T]
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-block_size:]# crop idx to the last block_size tokens
            # get prediction
            logits,loss=self(idx_cond) # call the forward(), logits [B,T,vocab_size]
            logits=logits[:,-1,:] #(B,vocab_size) last char of every blocks
            probs=F.softmax(logits,dim=-1)# sum(probability)==1
            idx_next=torch.multinomial(probs,num_samples=1)# get token according to probs
            idx=torch.cat((idx,idx_next),dim=1)
        return idx #(B,T+max_new_tokens)

model=BigramLanguageModel(vocab_size)
model=model.to(device)

# optimizer
optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)

import time
start_time = time.time()

for iter in range(max_iters+1):
    if iter % eval_interval == 0:
        losses=eval_loss()
        print(f"setp {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb,yb=get_batch('train')

    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if device == 'cuda':
        torch.cuda.synchronize()
    if iter % eval_interval == 0 and iter >0:
        elapsed = time.time() - start_time
        time_per_iter = elapsed / iter
        eta = (max_iters - iter) * time_per_iter

        print(f"iter {iter}/{max_iters} | loss {loss.item():.4f} | 剩余 {eta/60:.1f} min")

#parameters num
total_params = sum(p.numel() for p in model.parameters())
print(f"total_params is {total_params}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable_params is {trainable_params}")
# generate 
context=torch.zeros((1,1),dtype=torch.long,device=device)
print(decoder(model.generate(context,max_new_tokens=500)[0].tolist()))