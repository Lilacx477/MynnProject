from Mynn.autograd import Node
import numpy as np
from .module import Module
import ctypes
from numpy.ctypeslib import ndpointer
from pathlib import Path

#多重forループはim2col.dllで処理

dll_path = Path(__file__).resolve().parent.parent.parent / "c_files" / "im2col.dll"
lib = ctypes.CDLL(str(dll_path))

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

lib.col2im.argtypes = [
    ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
    ndpointer(dtype=np.float32, ndim=4, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib.col2im.restype = None

def just_conv(Kernels, X, padding, stride): #畳み込みの値計算
        
        X = np.ascontiguousarray(X, dtype=np.float32)
        Kernels = np.ascontiguousarray(Kernels, dtype=np.float32)
        N, C, H, W = X.shape
        kernels = Kernels.shape[0]
        channel = Kernels.shape[1]
        k_h = Kernels.shape[2]
        k_w = Kernels.shape[3]
        s_h, s_w = stride

        if padding=="same":
            p_h = k_h // 2
            p_w = k_w // 2
        else:
            p_h, p_w = padding
        H_out = (H + 2*p_h - k_h) // s_h + 1
        W_out = (W + 2*p_w - k_w) // s_w + 1
        
        X_p = np.zeros((N, C, H+2*p_h, W+2*p_w), dtype=np.float32)
        X_p[:, :, p_h:p_h + H, p_w:p_w + W] = X
        X_p = np.ascontiguousarray(X_p)
        H_p = H + 2 * p_h
        W_p = W + 2 * p_w

        if C != channel:
            raise ValueError("The number of input channels does not match the number of kernel channels")
        
        Kernels_flat = Kernels.reshape(kernels, k_h*k_w*C)
        X_p_flat = np.zeros((N*H_out*W_out, k_h*k_w*C), dtype=np.float32)

        lib.im2col(
            X_p,
            X_p_flat,
            N,
            H_p, W_p,
            H_out, W_out,
            C, k_h, k_w,
            s_h, s_w
        )

        KconvX_flat = Kernels_flat @ (X_p_flat).T #(kernel, N*H_out*W_out)の行列

        KconvX = KconvX_flat.reshape(kernels, N, H_out, W_out)

        KconvX = KconvX.transpose(1, 0, 2, 3) #(N, kernel, H_out, W_out)に変換

        return KconvX

def gradX_conv(grad, Kernels, X, padding, stride): #畳み込みのX偏微分関数
    X = np.ascontiguousarray(X, dtype=np.float32)
    Kernels = np.ascontiguousarray(Kernels, dtype=np.float32)
    grad = np.ascontiguousarray(grad, dtype=np.float32)
    N, C, H, W = X.shape
    kernels = Kernels.shape[0]
    channel = Kernels.shape[1]
    k_h = Kernels.shape[2]
    k_w = Kernels.shape[3]
    s_h, s_w = stride

    if padding=="same":
        p_h = k_h // 2
        p_w = k_w // 2
    else:
        p_h, p_w = padding
    H_out = (H + 2*p_h - k_h) // s_h + 1
    W_out = (W + 2*p_w - k_w) // s_w + 1
        
    X_p = np.zeros((N, C, H+2*p_h, W+2*p_w), dtype=np.float32)
    X_p[:, :, p_h:p_h + H, p_w:p_w + W] = X
    X_p = np.ascontiguousarray(X_p)
    H_p = H + 2 * p_h
    W_p = W + 2 * p_w

    if C != channel:
        raise ValueError("The number of input channels does not match the number of kernel channels")
        
    Kernels_flat = Kernels.reshape(kernels, k_h*k_w*C)

    grad = grad.transpose(0, 2, 3, 1)
    grad_flat = grad.reshape(N*H_out*W_out, kernels)

    dXp_flat = grad_flat @ Kernels_flat
    dXp_flat = np.ascontiguousarray(dXp_flat)

    dXp = np.zeros((N, C, H+2*p_h, W+2*p_w), dtype=np.float32)

    lib.col2im(
        dXp_flat,
        dXp,
        N,
        H_p, W_p,
        H_out, W_out,
        C, k_h, k_w,
        s_h, s_w
    )

    return dXp[:, :, p_h:p_h + H, p_w:p_w + W]

def gradF_conv(grad, Kernels, X, padding, stride): #畳み込みのF偏微分関数
    X = np.ascontiguousarray(X, dtype=np.float32)
    Kernels = np.ascontiguousarray(Kernels, dtype=np.float32)
    grad = np.ascontiguousarray(grad, dtype=np.float32)
    N, C, H, W = X.shape
    kernels = Kernels.shape[0]
    channel = Kernels.shape[1]
    k_h = Kernels.shape[2]
    k_w = Kernels.shape[3]
    s_h, s_w = stride

    if padding=="same":
        p_h = k_h // 2
        p_w = k_w // 2
    else:
        p_h, p_w = padding
    H_out = (H + 2*p_h - k_h) // s_h + 1
    W_out = (W + 2*p_w - k_w) // s_w + 1
        
    X_p = np.zeros((N, C, H+2*p_h, W+2*p_w), dtype=np.float32)
    X_p[:, :, p_h:p_h + H, p_w:p_w + W] = X
    X_p = np.ascontiguousarray(X_p)
    H_p = H + 2 * p_h
    W_p = W + 2 * p_w

    if C != channel:
        raise ValueError("The number of input channels does not match the number of kernel channels")
        
    X_p_flat = np.zeros((N*H_out*W_out, k_h*k_w*C), dtype=np.float32)

    lib.im2col(
        X_p,
        X_p_flat,
        N,
        H_p, W_p,
        H_out, W_out,
        C, k_h, k_w,
        s_h, s_w
    )

    grad = grad.transpose(0, 2, 3, 1)
    grad = np.ascontiguousarray(grad, dtype=np.float32)
    grad_flat = grad.reshape(N*H_out*W_out, kernels)

    dF_flat = grad_flat.T @ X_p_flat #(kernel, C*k_h*k_w)の行列
    dF_flat = np.ascontiguousarray(dF_flat, dtype=np.float32)

    dF = dF_flat.reshape(kernels, C, k_h, k_w)

    return dF

def convolution(Kernels, X, padding, stride): #畳み込み関数を定義
        Kernels = Node.to_node(Kernels)
        X = Node.to_node(X)
        kv = Kernels.value
        xv = X.value

        return Node(
        just_conv(kv, xv, padding=padding, stride=stride),
        [
            (lambda grad: gradF_conv(grad, kv, xv, padding=padding, stride=stride), Kernels),
            (lambda grad: gradX_conv(grad, kv, xv, padding=padding, stride=stride), X)
        ],
        just_conv(Kernels.tangent, xv, padding=padding, stride=stride) + just_conv(kv, X.tangent, padding=padding, stride=stride)
        )

def default_init_conv(cout, cin, kh, kw):
    return np.random.randn(cout, cin, kh, kw) * 0.01

def xavier_init_conv(cout, cin, kh, kw):
    fan_in = cin * kh * kw
    fan_out = cout * kh * kw
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(cout, cin, kh, kw) * std

def he_init_conv(cout, cin, kh, kw):
    fan_in = cin * kh * kw
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(cout, cin, kh, kw) * std

class Convolution(Module): #畳み込み層

    def __init__(self, kernels, kernel_size, input_shape, padding="same", stride=(1, 1), window_init=default_init_conv, bias_init=None):#kernel_sizeは(h,w)、input_shapeは(c,h,w)を想定
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.stride = stride

        self.Kernels = Node(window_init(kernels, input_shape[0], kernel_size[0], kernel_size[1]))
        
        if bias_init is None:
            self.b = Node(np.zeros((1, kernels, 1, 1), dtype=np.float32))
        else:
            self.b = Node(bias_init(kernels).reshape(1, kernels, 1, 1))

    def forward(self, X):
        return convolution(self.Kernels, X, padding=self.padding, stride=self.stride) + self.b