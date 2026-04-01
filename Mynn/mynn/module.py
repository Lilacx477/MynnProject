from Mynn.autograd.myautograd import Node
import numpy as np

class Module: #抽象モジュールクラス

    def __call__(self, *args): #Moduleが呼び出されたときにfowardを呼び出す
        return self.forward(*args)
    
    def parameters(self): #再帰的にModule内の全てのNodeを収集

        params = []
        visited = set() #多重探索防止

        def collect(obj):
            if isinstance(obj, Node):
                if id(obj) not in visited:
                    visited.add(obj)
                    params.append(obj) #Nodeならそのまま追加

            elif isinstance(obj, Module):
                for v in obj.__dict__.values():
                    collect(v) #Moduleならその属性に対し再帰的に探索
            
            elif isinstance(obj, list):
                for item in obj:
                    collect(item) #リストならその要素に対し再帰的に探索

            elif isinstance(obj, tuple):
                for item in obj:
                    collect(item) #タプルならその要素に対し再帰的に探索

            elif isinstance(obj, dict):
                for item in obj.values():
                    collect(item) #辞書ならその値に対し再帰的に探索

        for v in self.__dict__.values():
            collect(v)

        return params
    
    def zero_grad(self): #全てのNodeの勾配をゼロにする

        for node in self.parameters():
            node.grad = np.zeros_like(node.value)
    
    def forward(self, *args):
        raise NotImplementedError    
    
class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x