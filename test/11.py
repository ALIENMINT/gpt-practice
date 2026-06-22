import torch
x=torch.randn(2,3)
x.var
class BatchNorm1d:
    def __init__(self,dim,eps=1e-5):
        self.eps=eps
        self.gamma=torch.ones(dim)
        self.beta=torch.zeros(dim)
    def __call__(self, x):
        xmean=x.mean(1,keepdim=True)
        xvar=x.var(1,keepdim=True)
        xhat=(x-xmean)/torch.sqrt(xvar) # normalize
        self.out=self.gamma*xhat+self.beta # out = kx+b, strengthen the flexibilty of model
        return self.out
    def parameters(self):
        return [self.gamma,self.beta]