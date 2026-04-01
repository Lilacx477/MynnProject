from Mynn.autograd import sum
from Mynn.autograd import Node
from Mynn.mynn import Module

class L2Regularization(Module):
    def __init__(self, lam=1e-4, include_bias=False):
        self.lam = lam
        self.include_bias = include_bias

    def forward(self, model):
        reg = None

        for w in model.parameters():
            if (not self.include_bias) and w.value.ndim == 1:
                continue

            term = sum(w**2)

            if reg is None:
                reg = term
            else:
                reg = reg + term

        if  reg is None:
            return Node(0.0)
        
        return self.lam * reg
