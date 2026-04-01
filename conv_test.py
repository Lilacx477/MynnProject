import numpy as np
import time
from Mynn.autograd import Node
import Mynn.autograd as ad
import Mynn.datasets as ds
import Mynn.losses as ls
import Mynn.mynn as mn
import Mynn.optim as op
import numpy as np
import matplotlib.pyplot as plt


# あなたの実装に合わせて import を調整
from Mynn.autograd.myautograd import just_conv, gradF_conv, gradX_conv

def naive_conv(Kernels, X, padding="same", stride=1):
    """
    X: (C, H, W)
    Kernels: (K, C, Kh, Kw)
    return: (K, H_out, W_out)
    """
    X = np.asarray(X, dtype=np.float32)
    Kernels = np.asarray(Kernels, dtype=np.float32)

    C, H, W = X.shape
    K, Ck, Kh, Kw = Kernels.shape
    assert C == Ck

    if padding == "same":
        p_h = Kh // 2
        p_w = Kw // 2
    else:
        p_h = p_w = padding

    s_h = s_w = stride

    H_out = (H + 2 * p_h - Kh) // s_h + 1
    W_out = (W + 2 * p_w - Kw) // s_w + 1

    X_pad = np.zeros((C, H + 2 * p_h, W + 2 * p_w), dtype=np.float32)
    X_pad[:, p_h:p_h + H, p_w:p_w + W] = X

    Y = np.zeros((K, H_out, W_out), dtype=np.float32)

    for k in range(K):
        for h_out in range(H_out):
            for w_out in range(W_out):
                h_base = h_out * s_h
                w_base = w_out * s_w
                acc = 0.0
                for c in range(C):
                    for kh in range(Kh):
                        for kw in range(Kw):
                            acc += Kernels[k, c, kh, kw] * X_pad[c, h_base + kh, w_base + kw]
                Y[k, h_out, w_out] = acc
    return Y


def max_abs_diff(a, b):
    return float(np.max(np.abs(a - b)))


def rel_error(a, b, eps=1e-8):
    return float(np.max(np.abs(a - b) / np.maximum(eps, np.abs(a) + np.abs(b))))


def test_padding_stride_case(C, H, W, K, Kh, Kw, padding, stride, seed=0):
    np.random.seed(seed)

    X = np.random.randn(C, H, W).astype(np.float32)
    Kernels = np.random.randn(K, C, Kh, Kw).astype(np.float32)

    y_fast = just_conv(Kernels, X, padding=padding, stride=stride)
    y_naive = naive_conv(Kernels, X, padding=padding, stride=stride)

    print("=" * 72)
    print(f"Padding/Stride Test | X=({C},{H},{W}), K=({K},{C},{Kh},{Kw}), padding={padding}, stride={stride}")
    print("-" * 72)
    print("fast shape :", y_fast.shape)
    print("naive shape:", y_naive.shape)
    print("max abs diff:", max_abs_diff(y_fast, y_naive))
    print("rel error   :", rel_error(y_fast, y_naive))

    if y_fast.shape != y_naive.shape:
        raise AssertionError("shape mismatch")

    if not np.allclose(y_fast, y_naive, atol=1e-5, rtol=1e-4):
        raise AssertionError("value mismatch")

    print("PASS")


def run_padding_stride_tests():
    cases = [
        # padding=0, stride=1
        dict(C=2, H=5, W=6, K=3, Kh=3, Kw=3, padding=0,      stride=1, seed=0),

        # padding=same, stride=1
        dict(C=2, H=5, W=6, K=3, Kh=3, Kw=3, padding="same", stride=1, seed=1),

        # padding=0, stride=2
        dict(C=2, H=7, W=7, K=2, Kh=3, Kw=3, padding=0,      stride=2, seed=2),

        # padding=same, stride=2
        dict(C=2, H=7, W=7, K=2, Kh=3, Kw=3, padding="same", stride=2, seed=3),

        # 1x1 kernel, stride=2
        dict(C=3, H=8, W=8, K=4, Kh=1, Kw=1, padding=0,      stride=2, seed=4),

        # larger kernel
        dict(C=1, H=9, W=9, K=2, Kh=5, Kw=5, padding="same", stride=1, seed=5),
    ]

    for case in cases:
        test_padding_stride_case(**case)


if __name__ == "__main__":
    run_padding_stride_tests()