import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from digits import Network

net = Network(activation="relu")
data = np.load("digit_weights.npz")
n = len(net.W)
net.W = [data[f"arr_{i}"] for i in range(n)]
net.b = [data[f"arr_{i + n}"] for i in range(n)]

def preprocess(img):
    """Turn the 280x280 drawing into a 28x28 image the way MNIST digits are made."""
    arr = np.array(img)
    ys, xs = np.nonzero(arr > 20)
    if len(xs) == 0:
        return None
    # 1. crop to the digit
    crop = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    # 2. scale so the longest side is 20 pixels
    w, h = crop.size
    scale = 20.0 / max(w, h)
    crop = crop.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    # 3. paste into the middle of a 28x28 image
    small = Image.new("L", (28, 28), 0)
    small.paste(crop, ((28 - crop.size[0]) // 2, (28 - crop.size[1]) // 2))
    a = np.array(small, dtype=np.float32) / 255.0
    # 4. shift so the center of mass sits at the center
    total = a.sum()
    if total > 0:
        cy, cx = (np.indices(a.shape) * a).sum(axis=(1, 2)) / total
        a = np.roll(a, (int(round(13.5 - float(cy))), int(round(13.5 - float(cx)))), axis=(0, 1))
    return a

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Digit Recognizer")
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="black")
        self.canvas.pack()
        self.img = Image.new("L", (280, 280), 0)
        self.pen = ImageDraw.Draw(self.img)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.label = tk.Label(self.root, text="Draw a digit", font=("Arial", 18))
        self.label.pack()
        tk.Button(self.root, text="Predict", command=self.predict).pack()
        tk.Button(self.root, text="Clear", command=self.clear).pack()

    def paint(self, e):
        r = 10
        self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill="white", outline="white")
        self.pen.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)

    def predict(self):
        a = preprocess(self.img)
        if a is None:
            self.label.config(text="Draw a digit first")
            return
        probs = net.forward(a.reshape(1, 784))[0]
        top = np.argsort(probs)[::-1][:3]
        self.label.config(text="   ".join(f"{d}: {probs[d] * 100:.0f}%" for d in top))

    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (280, 280), 0)
        self.pen = ImageDraw.Draw(self.img)
        self.label.config(text="Draw a digit")

App().root.mainloop()