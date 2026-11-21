import os
import matplotlib.pyplot as plt
from digits import *

os.makedirs("results", exist_ok=True)

# ---------- Experiment 1: activation functions ----------
results = {}
for name in ["relu", "sigmoid", "tanh"]:
    print("Training with", name)
    net = Network(activation=name)
    results[name] = train(net, X_train, Y_train, y_train, epochs=10)

plt.figure()
for name, h in results.items():
    plt.plot(range(1, len(h["test_acc"]) + 1), h["test_acc"], label=name)
plt.xlabel("Epoch")
plt.ylabel("Test accuracy")
plt.title("Activation function comparison")
plt.legend()
plt.savefig("results/activation_comparison.png", dpi=150)
plt.close()

print("\nFinal test accuracy per activation:")
for name, h in results.items():
    print(f"  {name}: {h['test_acc'][-1]:.4f}")

# ---------- Experiment 2: overfitting and weight decay ----------
small_X, small_Y, small_y = X_train[:1000], Y_train[:1000], y_train[:1000]

print("\nTraining on 1000 images, no regularization")
plain = train(Network(), small_X, small_Y, small_y, epochs=100, lr=0.1)

print("\nTraining on 1000 images, with weight decay")
regularized = train(Network(), small_X, small_Y, small_y, epochs=100, lr=0.1, weight_decay=1e-3)

plt.figure()
plt.plot(plain["train_acc"], label="train, no regularization")
plt.plot(plain["test_acc"], label="test, no regularization")
plt.plot(regularized["test_acc"], label="test, with weight decay")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Overfitting and regularization")
plt.legend()
plt.savefig("results/overfitting.png", dpi=150)
plt.close()

print("\nOverfitting results (final epoch):")
print(f"  no regularization: train {plain['train_acc'][-1]:.4f}, test {plain['test_acc'][-1]:.4f}")
print(f"  with weight decay: train {regularized['train_acc'][-1]:.4f}, test {regularized['test_acc'][-1]:.4f}")