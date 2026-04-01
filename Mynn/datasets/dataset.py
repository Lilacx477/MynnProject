
import numpy as np

class Dataset: #抽象データセットクラス
    def __len__(self):
        raise NotImplementedError("please implement the __len__ method in the subclass")
    
    def __getitem__(self, idx):
        raise NotImplementedError("please implement the __getitem__ method in the subclass")
    
class TensorDataset(Dataset): #テンソルデータセットクラス
    def __init__(self, x, y = None):
        self.x = np.array(x, dtype=np.float32)
        self.y = None if y is None else np.array(y, dtype=np.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]