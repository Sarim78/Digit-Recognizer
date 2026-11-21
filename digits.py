import os
import gzip
import urllib.request
import numpy as np

# ---------- Step 1: load the data ----------
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

def download(filename, folder="mnist_data"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        return path
    for base in MIRRORS:
        try:
            print("Downloading", filename)
            urllib.request.urlretrieve(base + filename, path)
            return path
        except Exception as e:
            print("  failed on", base, "-", e)
    raise RuntimeError("Could not download " + filename)

def read_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 784).astype(np.float32) / 255.0

def read_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int64)

X_train = read_images(download(FILES["train_images"]))
y_train = read_labels(download(FILES["train_labels"]))
X_test = read_images(download(FILES["test_images"]))
y_test = read_labels(download(FILES["test_labels"]))

def one_hot(labels, k=10):
    out = np.zeros((len(labels), k), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1
    return out

Y_train = one_hot(y_train)

# ---------- Step 2: activation functions and softmax ----------
def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(np.float32)

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z): return np.tanh(z)
def tanh_grad(z): return 1 - np.tanh(z) ** 2

ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
}

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # subtract max so exp never overflows
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# ---------- Step 3: the network ----------
class Network:
    def __init__(self, sizes=(784, 128, 64, 10), activation="relu", seed=0):
        rng = np.random.default_rng(seed)
        self.act, self.act_grad = ACTIVATIONS[activation]
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def forward(self, X):
        self.zs = []
        self.As = [X]
        A = X
        last = len(self.W) - 1
        for i in range(len(self.W)):
            z = A @ self.W[i] + self.b[i]
            self.zs.append(z)
            A = softmax(z) if i == last else self.act(z)
            self.As.append(A)
        return A

    def loss(self, probs, Y, weight_decay=0.0):
        n = Y.shape[0]
        ce = -np.sum(Y * np.log(probs + 1e-12)) / n
        l2 = 0.5 * weight_decay * sum(np.sum(W ** 2) for W in self.W)
        return ce + l2

    def backward(self, Y, lr, weight_decay=0.0):
        n = Y.shape[0]
        delta = (self.As[-1] - Y) / n          # error at the output layer
        for i in reversed(range(len(self.W))):
            dW = self.As[i].T @ delta + weight_decay * self.W[i]
            db = delta.sum(axis=0)
            if i > 0:                           # pass the error to the previous layer
                delta = (delta @ self.W[i].T) * self.act_grad(self.zs[i - 1])
            self.W[i] -= lr * dW
            self.b[i] -= lr * db

# ---------- Step 4: training ----------
def accuracy(net, X, labels):
    return np.mean(np.argmax(net.forward(X), axis=1) == labels)

def train(net, X, Y, labels, epochs=10, lr=0.1, batch_size=64, weight_decay=0.0):
    history = {"train_acc": [], "test_acc": [], "loss": []}
    rng = np.random.default_rng(1)
    for epoch in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            net.forward(X[batch])
            net.backward(Y[batch], lr, weight_decay)
        probs = net.forward(X)
        history["loss"].append(net.loss(probs, Y, weight_decay))
        history["train_acc"].append(accuracy(net, X, labels))
        history["test_acc"].append(accuracy(net, X_test, y_test))
        print(f"epoch {epoch + 1}: loss {history['loss'][-1]:.4f}, "
              f"train {history['train_acc'][-1]:.4f}, test {history['test_acc'][-1]:.4f}")
    return history

if __name__ == "__main__":
    net = Network(activation="relu")
    train(net, X_train, Y_train, y_train, epochs=10)
    np.savez("digit_weights.npz", *net.W, *net.b)
    print("Saved weights to digit_weights.npz")