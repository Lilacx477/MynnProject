from Mynn.autograd.myautograd import Node
import numpy as np
from .module import Module
import ctypes
from numpy.ctypeslib import ndpointer
from pathlib import Path

dll_path = Path(__file__).resolve().parent.parent.parent / "c_files" / "im2col.dll"
lib = ctypes.CDLL(str(dll_path))

dll_path2 = Path(__file__).resolve().parent.parent.parent / "c_files" / "pool.dll"
lib2 = ctypes.CDLL(str(dll_path2))

lib.im2col.argtypes = [
    ndpointer(dtype=np.float32, ndim=4, flags="C_CONTIGUOUS"), #input
    ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), #output
    ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib.im2col.restype = None

lib2.maxpool.argtypes = [
    ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), #input
    ndpointer(dtype=np.float32, ndim=4, flags="C_CONTIGUOUS"), #output
    ndpointer(dtype=np.float32, ndim=4, flags="C_CONTIGUOUS"), #index
    ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib2.maxpool.restype = None

lib2.avepool.argtypes = [
    ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), #input
    ndpointer(dtype=np.float32, ndim=4, flags="C_CONTIGUOUS"), #output
    ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib2.avepool.restype = None

lib2.grad_maxpool.argtypes = [
    ndpointer(dtype=np.float32, ndim=4, flags="C_CONTIGUOUS"), #input
    ndpointer(dtype=np.float32, ndim=4, flags="C_CONTIGUOUS"), #output
    ndpointer(dtype=np.float32, ndim=4, flags="C_CONTIGUOUS"), #index
    ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib2.grad_maxpool.restype = None

lib2.grad_avepool.argtypes = [
    ndpointer(dtype=np.float32, ndim=4, flags="C_CONTIGUOUS"), #input
    ndpointer(dtype=np.float32, ndim=4, flags="C_CONTIGUOUS"), #output
    ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib2.grad_maxpool.restype = None

lib2.tangent_maxpool.argtypes = [
    ndpointer(dtype=np.float32, ndim=4, flags="C_CONTIGUOUS"), #input
    ndpointer(dtype=np.float32, ndim=4, flags="C_CONTIGUOUS"), #output
    ndpointer(dtype=np.float32, ndim=4, flags="C_CONTIGUOUS"), #index
    ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib2.tangent_maxpool.restype = None

def just_maxpool(X, window_shape, output_shape):

    N, C, H, W = X.shape
    H_win, W_win = window_shape
    H_out, W_out = output_shape

    X = np.ascontiguousarray(X,dtype=np.float32)
    X_col = np.zeros((N*H_out*W_out, H_win*W_win*C),dtype= np.float32)

    lib.im2col(
        X,
        X_col,
        N,
        H, W,
        H_out, W_out,
        C, H_win, W_win,
        H_win, W_win
    )

    PoolX = np.zeros((N, C, H_out, W_out), dtype=np.float32)
    Idx = np.zeros((N, C, H_out, W_out), dtype=np.float32)

    lib2.maxpool(
        X_col,
        PoolX,
        Idx,
        N, C, H, W,
        H_out, W_out,
        H_win, W_win
    )

    return PoolX, Idx

def grad_maxpool(grad, X_shape, Idx):

    grad = np.ascontiguousarray(grad, dtype=np.float32)
    Idx = np.ascontiguousarray(Idx, dtype=np.float32)
    
    N, C, H, W = X_shape
    H_out = grad.shape[2]
    W_out = grad.shape[3]

    dX = np.zeros((N, C, H, W), dtype=np.float32)

    lib2.grad_maxpool(
        grad,
        dX,
        Idx,
        N,
        C, H, W,
        H_out, W_out
    )

    return dX

def tangent_maxpool(Tangent, Idx):

    Tangent = np.ascontiguousarray(Tangent, dtype=np.float32)
    Idx = np.ascontiguousarray(Idx, dtype=np.float32)
    
    N, C, H, W = Tangent.shape
    H_out = Idx.shape[2]
    W_out = Idx.shape[3]

    Tan = np.zeros((N, C, H_out, W_out), dtype=np.float32)

    lib2.tangent_maxpool(
        Tangent,
        Tan,
        Idx,
        N,
        C, H, W,
        H_out, W_out
    )

    return Tan

def just_avepool(X, window_shape, output_shape):

    N, C, H, W = X.shape
    H_win, W_win = window_shape
    H_out, W_out = output_shape

    X = np.ascontiguousarray(X)
    X_col = np.zeros((H_out*W_out, H_win*W_win*C), dtype=np.float32)

    lib.im2col(
        X,
        X_col,
        N,
        H, W,
        H_out, W_out,
        C, H_win, W_win,
        H_win, W_win
    )

    PoolX = np.zeros((N,C, H_out, W_out), dtype=np.float32)

    lib2.avepool(
        X_col,
        PoolX,
        N,
        C, H_out, W_out,
        H_win, W_win
    )

    return PoolX

def grad_avepool(grad, X_shape, window_shape):

    grad = np.ascontiguousarray(grad, dtype=np.float32)
    
    C, H, W = X_shape
    H_out = grad.shape[1]
    W_out = grad.shape[2]
    H_win, W_win = window_shape

    dX = np.zeros((C, H, W), dtype=np.float32)

    lib2.grad_avepool(
        grad,
        dX,
        C, H, W,
        H_out, W_out,
        H_win, W_win
    )

    return dX

def tangent_avepool(Tangent, out_shape, window_shape):

    Tangent = np.ascontiguousarray(Tangent, dtype=np.float32)
    
    C, H, W = Tangent.shape
    H_out, W_out = out_shape
    H_win, W_win = window_shape

    Tan = np.zeros((C, H_out, W_out), dtype=np.float32)
    Tangent_col = np.zeros((H_out*W_out,C*H_win*W_win ), dtype=np.float32)

    lib.im2col(
        Tangent,
        Tangent_col,
        H, W,
        W_out, H_out,
        C, H_win, W_win,
        H_win, W_win
    )

    lib2.avepool(
        Tangent_col,
        Tan,
        C, H_out, W_out,
        H_win, W_win
    )

    return Tan

def maxpooling(X, window_shape, output_shape):
    PoolX, Idx = just_maxpool(X.value, window_shape, output_shape)
    return Node(
        PoolX,
        [
            (lambda grad: grad_maxpool(grad, X.value.shape, Idx), X)
        ],
        tangent_maxpool(X.tangent, Idx)
        )

def avepooling(X, window_shape, output_shape):
    PoolX = just_avepool(X.value, window_shape, output_shape)
    return Node(
        PoolX,
        [
            (lambda grad: grad_avepool(grad, X.shape, window_shape), X)
        ],
        tangent_avepool(X.tangent, output_shape,  window_shape)
        )

class MaxPooling(Module):
    
    def __init__(self, input_shape, window_shape, output_shape=None):
        self.C, self.H, self.W = input_shape
        if (window_shape == "adaptive") and (output_shape is not None):
            self.oH, self.oH = output_shape
            self.wH = self.H // self.oH
            self.wW = self.W // self.oW
        elif window_shape == "adaptive":
            raise TypeError("need output_shape when you use adaptive-mode")
        else:
            self.wH, self.wW = window_shape
            self.oH = self.H // self.wH
            self.oW = self.W // self.wW

    def forward(self, X):
        return maxpooling(X, (self.wH, self.wW), (self.oH, self.oW))


class AveragePooling(Module):
    
    def __init__(self, input_shape, window_shape, output_shape=None):
        self.C, self.H, self.W = input_shape
        if (window_shape == "adaptive") and (output_shape is not None):
            self.oH, self.oH = output_shape
            self.wH = self.H // self.oH
            self.wW = self.W // self.oH
        elif window_shape == "adaptive":
            raise TypeError("need output_shape when you use adaptive-mode")
        else:
            self.oH = self.H // self.wH
            self.oW = self.W // self.wW
    
    def forward(self, X):
        return avepooling(X, (self.wH, self.wW), (self.oH, self.oW))