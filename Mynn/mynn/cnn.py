from .module import Module, Sequential
from .linear import *
from .convolution import Convolution
from .pooling import MaxPooling, AveragePooling
from .flatten import Flatten
from .activations import ReLU, Softmax
from .convolution import *

class CNN(Module):
    def __init__(self, input_size, conv_arguments , hidden_sizes, output_size): #argは(kernel数, (kh, kw),(ph, pw) (sh, sw))

        self.layers = []

        C, H, W = input_size

        for i in range(len(conv_arguments)):
            
            kernel = conv_arguments[i][0]
            kh, kw = conv_arguments[i][1]
            padding = conv_arguments[i][2]

            if padding == "same":
                ph = kh // 2
                pw = kw // 2
            else:
                ph, pw = padding

            sh, sw = conv_arguments[i][3]


            self.layers.append(Convolution(kernel, (kh, kw), (C, H, W), padding=(ph, pw), stride=(sh, sw), window_init=he_init_conv)) #畳み込み層を追加

            H_out = (H + 2*ph - kh) // sh + 1
            W_out = (W + 2*pw - kw) // sw + 1

            self.layers.append(ReLU()) #活性化関数を追加
            self.layers.append(MaxPooling((kernel, H_out, W_out), (2, 2))) #プーリング層を追加

            H = H_out // 2
            W = W_out // 2
            C = kernel

        self.layers.append(Flatten((C, H, W))) #Flatten層を追加
        size_of_flatten_output = C * H * W

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        sizes = [size_of_flatten_output] + list(hidden_sizes) + [output_size]

        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i],sizes[i+1],weight_init=he_init)) #線形層を追加

            if i < len(sizes) - 2:
                self.layers.append(ReLU()) #活性化関数を追加(出力層は除く)
        self.layers.append(Softmax())

        self.networks = Sequential(*self.layers) #Sequentialモジュールにまとめる

    def forward(self, x):
        z = self.networks(x)
        return z