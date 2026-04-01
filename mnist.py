import Mynn.autograd as ad
import Mynn.datasets as ds
import Mynn.losses as ls
import Mynn.mynn as mn
import Mynn.optim as op
import Mynn.mynn.save_load as sl
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import gzip
import struct
from pathlib import Path

def load_mnist_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid magic number in MNIST image file: {}".format(magic))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_mnist_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid magic number in MNIST label file: {}".format(magic))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

x_train_path = Path(__file__).resolve().parent / "MNIST" / "train-images-idx3-ubyte.gz"
y_train_path = Path(__file__).resolve().parent / "MNIST" / "train-labels-idx1-ubyte.gz"
x_test_path = Path(__file__).resolve().parent / "MNIST" / "t10k-images-idx3-ubyte.gz"
y_test_path = Path(__file__).resolve().parent / "MNIST" / "t10k-labels-idx1-ubyte.gz"

x_train = load_mnist_images(x_train_path)
y_train = load_mnist_labels(y_train_path)
x_test = load_mnist_images(x_test_path)
y_test = load_mnist_labels(y_test_path)

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

x_train = x_train[:, np.newaxis, :, :]
x_test = x_test[:, np.newaxis, :, :]

def to_one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
    one_hot[np.arange(labels.size), labels] = 1.0
    return one_hot

y_train_one_hot = to_one_hot(y_train)

dataset = ds.TensorDataset(x_train, y_train_one_hot)

loader = ds.DataLoader(dataset, batch_size=64, shuffle=True)

model = mn.CNN(input_size=(1, 28, 28), conv_arguments=[(8, (3, 3), "same", (1, 1))], hidden_sizes=128, output_size=10)

optimizer = op.SGD(model.parameters(), lr=0.05)

loss_history = []


criterion = ls.CROSS_ENTROPY()
regularization = ls.L2Regularization(lam=0)

for epoch in range(10):
    print("Epoch", epoch, "start")
    for xb, yb in loader:
        model.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb) + regularization(model)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.value))
    print("Epoch", epoch, "Loss:", loss.value)

sl.save_model(model,
            "models/mnist_cnn_20260401.npz",
            config={"input_size": (1, 28, 28),
                    "conv_arguments": [(8, (3, 3),
                    "same", (1, 1))],
                    "hidden_sizes": 128,
                    "output_size": 10})

def moving_average(x, window=50):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

smoothed = moving_average(loss_history, window=100)

plt.plot(smoothed)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Training Loss")
plt.show()

correct = 0
total = 0

for xb, yb in ds.DataLoader(ds.TensorDataset(x_test, y_test), batch_size=64, shuffle=False):
    pred = model(xb)

    # 予測ラベル
    pred_label = np.argmax(pred.value, axis=1)

    # 正解ラベル
    true_label = yb.value if hasattr(yb, "value") else yb

    correct += np.sum(pred_label == true_label)
    total += len(true_label)

accuracy = correct / total
print("Test Accuracy:", accuracy)