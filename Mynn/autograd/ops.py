# 各種数学関数

import numpy as np
from .myautograd import Node

def abs(node):
    node = Node.to_node(node)
    grad = np.sign(node.value)
    return Node(np.abs(node.value), [(grad, node)], grad * node.tangent)

def sin(node):
    node = Node.to_node(node)
    return Node(np.sin(node.value), [(np.cos(node.value), node)], np.cos(node.value) * node.tangent)

def cos(node):
    node = Node.to_node(node)
    return Node(np.cos(node.value), [(-np.sin(node.value), node)], -np.sin(node.value) * node.tangent)

def exp(node, base=np.e):
    node = Node.to_node(node)
    return Node(base**node.value, [(np.log(base) * (base**node.value), node)], np.log(base) * (base**node.value) * node.tangent)

def log(node, base=np.e):
    node = Node.to_node(node)
    return Node(np.log(node.value)/np.log(base), [(1/(node.value * np.log(base)), node)], (1/(node.value * np.log(base))) * node.tangent)

def sum(node, axis=None, keepdims=False):
    node = Node.to_node(node)
    value = np.sum(node.value, axis=axis, keepdims=keepdims)

    def grad_fn(grad):
        g = grad

        if axis is None:
            return np.ones_like(node.value) * g

        axes = axis if isinstance(axis, tuple) else (axis,)
        axes = tuple(a if a >= 0 else a + node.value.ndim for a in axes)

        if not keepdims:
            for ax in sorted(axes):
                g = np.expand_dims(g, axis=ax)

        return np.ones_like(node.value) * g

    tangent = np.sum(node.tangent, axis=axis, keepdims=keepdims)
    return Node(value, [(grad_fn, node)], tangent)

def mean(node, axis=None, keepdims=False):
    node = Node.to_node(node)
    value = np.mean(node.value, axis=axis, keepdims=keepdims)

    if axis is None:
        count = node.value.size
    else:
        axes = axis if isinstance(axis, tuple) else (axis,)
        axes = tuple(a if a >= 0 else a + node.value.ndim for a in axes)
        count = 1
        for ax in axes:
            count *= node.value.shape[ax]

    def grad_fn(grad):
        g = grad

        if axis is None:
            return np.ones_like(node.value) * (g / count)

        if not keepdims:
            for ax in sorted(axes):
                g = np.expand_dims(g, axis=ax)

        return np.ones_like(node.value) * (g / count)

    tangent = np.mean(node.tangent, axis=axis, keepdims=keepdims)
    return Node(value, [(grad_fn, node)], tangent)

def relu(node):
    node = Node.to_node(node)
    value = np.maximum(0, node.value)
    grad = (node.value > 0).astype(np.float64)
    return Node(value, [(grad, node)], grad * node.tangent)

def sigmoid(node):
    node = Node.to_node(node)
    value = 1/(1+np.exp(-node.value))
    grad = value * (1- value)
    return Node(value, [(grad, node)], grad * node.tangent)

def tanh(node):
    node = Node.to_node(node)
    value = np.tanh(node.value)
    grad = 1 - value**2
    return Node(value, [(grad, node)], grad * node.tangent)

def softmax(node, axis=-1):
    node = Node.to_node(node)

    shifted = node.value - np.max(node.value, axis=axis, keepdims=True)
    ex = np.exp(shifted)
    value = ex / np.sum(ex, axis=axis, keepdims=True)

    def grad_fn(grad):
        dot = np.sum(grad * value, axis=axis, keepdims=True)
        return value * (grad - dot)

    tangent = grad_fn(node.tangent)
    return Node(value, [(grad_fn, node)], tangent)

def mse(pred, target): #平均二乗誤差
    pred = Node.to_node(pred)
    target = Node.to_node(target)
    diff = pred - target
    return sum(diff**2)/diff.value.size

def mae(pred, target): #平均絶対誤差
    pred = Node.to_node(pred)
    target = Node.to_node(target)
    diff = pred - target
    return sum(abs(diff))/diff.value.size

def huber(pred, target, delta=1.0): #Huber損失
    pred = Node.to_node(pred)
    target = Node.to_node(target)
    diff = pred - target
    mask = (abs(diff) <= delta).value
    squared = 0.5 * diff**2
    linear = delta * (abs(diff) - 0.5 * delta)
    return sum(mask * squared + (1 - mask) * linear )/diff.value.size

def cross_entropy(pred, target): #クロスエントロピー誤差
    pred = Node.to_node(pred) #predは確率分布と仮定
    target = Node.to_node(target) #targetはone-hot-encodedと仮定
    eps = 1e-12
    return -sum(target * log(pred+eps))/target.value.shape[0] #バッチ平均を取っている