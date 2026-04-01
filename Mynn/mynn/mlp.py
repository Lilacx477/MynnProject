from .module import Module, Sequential
from .linear import *
from .activations import ReLU, Sigmoid

class MLP(Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.layers = []

        sizes = [input_size] + list(hidden_sizes) + [output_size]

        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i],sizes[i+1],weight_init=he_init)) #線形層を追加

            if i < len(sizes) - 2:
                self.layers.append(ReLU()) #活性化関数を追加(出力層は除く)
        self.layers.append(Sigmoid())

        self.networks = Sequential(*self.layers) #Sequentialモジュールにまとめる

    def forward(self, x):
        return self.networks(x)
