import torch

# activate function
class Swish(nn.Module):
    def __init__(self,beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self,x):
        return x*torch.sigmoid(self.beta*x)
