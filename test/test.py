with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
    # print(len(text))
    # print(text[:100])
    #print(''.join(sorted(list(set(text)))))
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi={ch:i for i,ch in enumerate(chars)}
    # print(chars)
    # print(stoi)
    itos = {i:ch for i,ch in enumerate(chars)}
    # print(itos)
    encode = lambda s: [stoi[c] for c in s]
    #print(encode)
    decode = lambda l: ''.join(itos[i] for i in l)
    #print(decode)
    # print(encode("hello world!"))
    # print(decode(encode("hello world!")))

    import torch 
    data = torch.tensor(encode(text),dtype=torch.long)
    #print(data.shape)
    # print(data[:1000])

    n = int(0.9*len(data))
    train_data=data[:n]
    val_data=data[n:]

    # block_size=8
    # print(train_data[:block_size+1])
    # x=train_data[:block_size]
    # y=train_data[1:block_size+1] # offset 1
    # for i in range(block_size):
    #     context = x[:i+1]
    #     target=y[i]
    #     print(f"input is {context}, the target is {target}")
    
    torch.manual_seed(1337)
    batch_size = 4
    block_size = 8

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data)-block_size,(batch_size,))
        x=torch.stack([data[i:i+block_size] for i in ix])
        y=torch.stack([data[i+1:i+block_size+1] for i in ix]) # offset 1
        # print(x)
        # print(y)
        return x,y
    xb,yb=get_batch(split='train') # torch.Size[4,8]
    for b in range(batch_size):
        for t in range(block_size):
            context=xb[b,:t+1]
            target=yb[b,t]
            # print(f"{context.tolist()} => {target}")
    
    import torch.nn as nn
    from torch.nn import functional as F
    torch.manual_seed(1337)

    class BigramLanguageModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size,vocab_size) # create a (vocab_size, n_emb) table

        def forward(self,idx,targets=None): # (,[B,T],[B,T])
            logits =self.token_embedding_table(idx) # search table by idx elems, logits.shape is [B,T,n_emb]

            if targets is None:
                loss = None
            else:
                B,T,C= logits.shape
                logits=logits.view(B*T,C)
                targets=targets.view(B*T)
                loss = F.cross_entropy(logits,targets)
            return logits,loss
        
        def generate(self,idx,max_new_tokens):
            for _ in range(max_new_tokens):
                logits,loss = self(idx)
                logits=logits[:,-1,:] # (B,T,C) => (B,C), the last data of each batch is held, is data not tensor
                probs= F.softmax(logits,dim=-1) # (B,C)
                idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
                idx = torch.cat((idx,idx_next),dim=1) #(B,T+1)
            return idx
        
    m = BigramLanguageModel(vocab_size=vocab_size)
    out,loss =m(xb,yb)
    # print(out.shape)
    # print(loss)
    # print(decode(m.generate(torch.zeros((1,1),dtype=torch.long), max_new_tokens=100)[0].tolist()))

    optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)

    batch_size=32
    for steps in range(10000):
        xb,yb=get_batch('train')
        logits,loss=m(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(loss.item())
    print(decode(m.generate(torch.zeros((1,1),dtype=torch.long), max_new_tokens=100)[0].tolist()))