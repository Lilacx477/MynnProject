# Optimizer

from Mynn.autograd import Node
import numpy as np

class Optimizer: #抽象オプティマイザクラス
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError("please implement the step method in the subclass")
    
    def zero_grad(self):
        for node in self.parameters:
            node.grad[:] = 0

class SGD(Optimizer): #確率的勾配降下法
    def step(self):
        for node in self.params:
            node.value -= self.lr * node.grad

class SGDM(Optimizer): #モメンタム付きSGD
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params, lr) #Optimizerから継承して初期化
        self.momentum = momentum
        self.v = [np.zeros_like(node.value) for node in self.params]  #各ノードのモメンタムを初期化

    def step(self):
        for i, node in enumerate(self.params):
            self.v[i] = self.momentum * self.v[i] - self.lr * node.grad #モメンタム更新
            node.value += self.v[i]

class ADAMW(Optimizer): #AdamW
    def __init__(self, params, lr=0.01, a=0.9, b=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr) #Optimizerから継承して初期化
        self.a = a
        self.b = b
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.v = [np.zeros_like(node.value) for node in self.params] #2次モーメント初期化
        self.m = [np.zeros_like(node.value) for node in self.params] #1次モーメント初期化

    def step(self):
        self.t += 1
        for i, node in enumerate(self.params):
            self.v[i] = self.a * self.v[i] + (1-self.a) * (node.grad)**2
            self.m[i] = self.b * self.m[i] + (1-self.b) * node.grad
            e_v = self.v[i] / (1-self.a**self.t)
            e_m = self.m[i] / (1-self.b**self.t)
            node.value = node.value  - self.lr * e_m / (np.sqrt(e_v) + self.eps) -self.lr * self.weight_decay * node.value