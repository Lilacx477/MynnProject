from ..autograd.myautograd import Node
import numpy as np
from .module import Module

import ctypes
from numpy.ctypeslib import ndpointer
from pathlib import Path

dll_path = Path(__file__).resolve().parent.parent.parent / "c_files" / "im2col.dll"
lib = ctypes.CDLL(str(dll_path))

def flatten(X):
    xv = X.value
    N, C, H, W = xv.shape
    xgrad_flat = X.grad
    xtan = X.tangent

    xv_flat = xv.reshape(N, C*H*W)

    xtan_flat = xtan.reshape(N, C*H*W)

    def grad_fn(grad):
        return grad.reshape(N, C, H, W)

    return Node(xv_flat, [(grad_fn, X)], xtan_flat)

class Flatten(Module):
    
    def __init__(self, input_shape): #input_shapeは(N, C, H, W)を想定
        super().__init__()
        self.input_shape = input_shape
    
    def forward(self, X):

        return flatten(X)