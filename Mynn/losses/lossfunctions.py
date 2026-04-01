#　損失関数

from Mynn.autograd import Node
import Mynn.autograd as ad
import numpy as np
from Mynn.mynn import Module 

class MSE(Module): #MSE
    def forward(self, pred, target):
        return ad.mse(pred, target)

class MAE(Module): #MAE
    def forward(self, pred, target):
        return ad.mae(pred, target)
    
class HUBER(Module): #Huber
    def forward(self, pred, target):
        return ad.huber(pred, target)
    
class CROSS_ENTROPY(Module): #クロスエントロピー
    def forward(self, pred, target):
        return ad.cross_entropy(pred, target)
