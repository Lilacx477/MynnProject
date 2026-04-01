from Mynn.autograd import Node
import numpy as np
from .module import Module

def default_init(in_features, out_features): #単純な正規分布による初期化
    return np.random.randn(in_features, out_features) * 0.01

def xavier_init(in_features, out_features): #Xavier初期化
    std = np.sqrt(2 / (in_features + out_features))
    return np.random.randn(in_features, out_features) * std

def he_init(in_features, out_features): #He初期化
    std = np.sqrt(2 / in_features)
    return np.random.randn(in_features, out_features) * std


class Linear(Module): #線形層

    def __init__(self, in_features, out_features, weight_init=default_init, bias_init=None):

        self.in_features = in_features
        self.out_features = out_features

        self.W = Node(weight_init(in_features, out_features))

        if bias_init is None:
            self.b = Node(np.zeros(out_features))
        else:
            self.b = Node(bias_init(out_features))
    
    def forward(self, x):
        return x @ self.W + self.b