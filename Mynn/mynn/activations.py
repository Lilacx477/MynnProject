import Mynn.autograd as ad
from .module import Module
import numpy as np

class ReLU(Module): #ReLU活性化関数
    def forward(self, node):
        return ad.relu(node)
    
class Sigmoid(Module): #Sigmoid活性化関数
    def forward(self, node):
        return ad.sigmoid(node)
    
class Tanh(Module): #Tanh活性化関数
    def forward(self, node):
        return ad.tanh(node)
    
class Softmax(Module): #Softmax活性化関数
    def forward(self, node):
        return ad.softmax(node)