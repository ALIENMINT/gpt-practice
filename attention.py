import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
B,T,C=4,8,32
x=torch.randn(B,T,C)
xbow=torch.zeros(B,T,C)
# print(xbow)
for b in range(B):
    for t in range(T):
        xprev=x[b,:t+1]
        # print(xprev)
        xbow[b,t]=torch.mean(xprev,0)
        # print(xbow)
# print(xbow)

# torch.manual_seed(42)
# a=torch.tril(torch.ones(3,3))
# a=a/torch.sum(a,1,keepdim=True)
# b=torch.randint(0,10,(3,2)).float()
# c=a@b
# dbow=torch.zeros(3,2)
# for i in range(3):
#     for j in range(2):
#         dp=b[i,:j+1]
#         dbow[i,j]=torch.mean(dp,0)
# print(f"b={b}")
# print(f"c={c}")
# print(f"d={dbow}")

# wei = torch.tril(torch.ones(T,T))
# wei=wei/wei.sum(1,keepdim=True)
# xbow2=wei@x # (T,T)@(B,T,C) -->(extended) (B,T,T)@(B,T,C) = (B,T,C)
# print(torch.allclose(xbow, xbow2, atol=1e-6))
# print(torch.abs(xbow - xbow2).max()) 

# self-attention
head_size =16
key=nn.Linear(C,head_size,bias=False)
query=nn.Linear(C,head_size,bias=False)
k=key(x)
q=query(x)

wei=q@k.transpose(-2,-1) #(B,T,16) @ (B,16,T) => (B,T,T)
print(f"wei table ={wei[:1,:,:]}")

tril=torch.tril(torch.ones(T,T))
wei=wei.masked_fill(tril==0,float('-inf'))
print(f"wei inf ={wei[:1,:,:]}")

wei=F.softmax(wei,dim=-1)
print(f"wei softmax ={wei[:1,:,:]}")

out=wei@x
# print(out)