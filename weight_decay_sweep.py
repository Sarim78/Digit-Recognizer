import io
import contextlib
import matplotlib.pyplot as plt
from digits import *

small_X, small_Y, small_y = X_train[:1000], Y_train[:1000], y_train[:1000]

values = [0.0, 1e-3, 1e-2, 3e-2, 1e-1]
curves = {}

for wd in values:
    with contextlib.redirect_stdout(io.StringIO()):  # hides the per-epoch lines
        h = train(Network(), small_X, small_Y, small_y, epochs=100, lr=0.1, weight_decay=wd)
    curves[wd] = h
    print(f"weight decay {wd}: train {h['train_acc'][-1]:.4f}, test {h['test_acc'][-1]:.4f}")

plt.figure()
for wd, h in curves.items():
    plt.plot(h["test_acc"], label=f"weight decay {wd}")
plt.xlabel("Epoch")
plt.ylabel("Test accuracy")
plt.title("Effect of weight decay strength (1,000 training images)")
plt.legend()
plt.savefig("results/weight_decay_sweep.png", dpi=150)
plt.close()
print("Saved results/weight_decay_sweep.png")