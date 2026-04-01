import numpy as np
from Mynn.autograd import Node
from Mynn.autograd import cross_entropy
from Mynn.mynn.cnn import CNN
from Mynn.mynn.convolution import *

N = 4
x = np.random.randn(N, 1, 28, 28).astype(np.float32)
x = Node.to_node(x)

t = np.zeros((N, 10), dtype=np.float32)
labels = np.array([1, 3, 2, 0])
t[np.arange(N), labels] = 1.0
t = Node.to_node(t)

model = CNN(
    input_size=(1, 28, 28),
    conv_arguments=[
        (8, (3, 3), "same", (1, 1)),
        (16, (3, 3), "same", (1, 1)),
    ],
    hidden_sizes=64,
    output_size=10
)

print("layers", model.layers)

y= model(x)
print("output shape:", y.value.shape)
print("row sum:", np.sum(y.value, axis=1))

loss = cross_entropy(y, t)
print("loss:", loss.value)

loss.backward()
print("backward ok")