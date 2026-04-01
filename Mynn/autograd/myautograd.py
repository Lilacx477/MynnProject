#自動微分のためのクラスや演算子オーバーロード
import numpy as np

class Node: #ノードクラス
    def __init__(self, value, children=None, tangent=None):
        self.value = np.array(value, dtype=np.float32)
        self.grad = np.zeros_like(self.value)
        self.children = children or []
        if tangent is None:
            self.tangent = np.zeros_like(self.value)
        else:
            self.tangent = np.array(tangent, dtype=np.float32)

    def __repr__(self): #表現の指定
        return f"Node(value={self.value}, grad={self.grad})"
    
    def __hash__(self): #ハッシュ関数の明示
        return id(self)
    
    @staticmethod
    def to_node(x): #定数をノード化
        if isinstance(x, Node):
            return x
        else:
            return Node(x)
    @staticmethod    
    def unbroadcast(grad, shape): #アンブロードキャスト
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)

        for i in reversed(range(len(shape))):
            if shape[i] ==1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    
    @staticmethod
    def matmul(a,b): #行列積を関数として定義(後で数値のときと区別するために局所勾配を関数で渡す)
        a = Node.to_node(a)
        b = Node.to_node(b)
        return Node(a.value @ b.value, [(lambda grad: grad @ b.value.T, a), (lambda grad: a.value.T @ grad, b)], a.tangent @ b.value + a.value @ b.tangent)

    #演算子オーバーロード
    def __add__(self, other):
        other = Node.to_node(other)
        if isinstance(other, Node):
            return Node(self.value + other.value, [(1,self), (1,other)], self.tangent + other.tangent)
    
    def __sub__(self, other):
        other = Node.to_node(other)
        if isinstance(other, Node):
            return Node(self.value - other.value, [(1,self), (-1,other)], self.tangent - other.tangent)

    def __neg__(self):
        return Node(-self.value, [(-1,self)], -self.tangent)
    
    def __mul__(self, other):
        other = Node.to_node(other)
        if isinstance(other, Node):
            return Node(self.value * other.value, [(other.value, self), (self.value, other)], self.tangent * other.value + self.value * other.tangent)
    
    def __matmul__(self, other):
        return Node.matmul(self, other)
    
    def __truediv__(self, other):
        other = Node.to_node(other)
        if isinstance(other, Node):
            return Node(self.value / other.value, [(1/other.value, self), (-self.value/(other.value**2), other)], self.tangent/other.value - self.value*(1/other.value**2)*other.tangent)

    def __pow__(self, power):
        return Node(self.value ** power, [(power * self.value ** (power - 1),self)], self.tangent * power * self.value ** (power - 1))
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return Node.to_node(other)-self
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return Node.to_node(other)/self

    #DFSでトポロジカルソートされたリストを作成して逆伝播
    def backward(self):
        topo=[]
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for _, child in node.children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = np.ones_like(self.value)

        for node in reversed(topo):
            for local_grad , child in node.children:
                if callable(local_grad): #局所勾配が関数のときはそれを適用、数値のときはそのまま積を取る
                    grad = local_grad(node.grad)
                else:
                    grad = local_grad * node.grad

                grad = Node.unbroadcast(grad, child.value.shape)
                child.grad += grad