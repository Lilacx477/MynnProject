import numpy as np
from Mynn.autograd import Node

class DataLoader: #データローダークラス
    def __init__(self, dataset, batch_size=32, shuffle=True, tonode=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tonode = tonode

    def __iter__(self):
        indices = np.arange(len(self.dataset)) #添え字を生成
        if self.shuffle: #シャッフル
            np.random.shuffle(indices)

        for s in range(0, len(indices), self.batch_size):
            batch_indices = indices[s:s+self.batch_size] #バッチの添え字を取得
            batch = self.dataset[batch_indices] #バッチのデータを取得
            if isinstance(batch, tuple): #(input, target)のタプル構造か判定
                x, y = batch
                if self.tonode:
                    yield Node.to_node(x), Node.to_node(y)
                else: yield x, y 

            else: #入力データのみの場合
                x = batch
                if self.tonode: #Nodeで返すかどうか
                    yield Node.to_node(x)
                else:
                    yield x

    def __len__(self):
        return np.ceil(len(self.dataset) / self.batch_size) #バッチ数を返す(全データを使い切るまでのループ回数)