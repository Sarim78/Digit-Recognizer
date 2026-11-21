import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from digits import Network

net = Network(activation="relu")
data = np.load("digit_weights.npz")
n = len(net.W)
net.W = [data[f"arr_{i}"] for i in range(n)]
net.b = [data[f"arr_{i + n}"] for i in range(n)]

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
        small = self.img.resize((28, 28), Image.LANCZOS)
        x = np.array(small, dtype=np.float32).reshape(1, 784) / 255.0
        probs = net.forward(x)[0]
        top = np.argsort(probs)[::-1][:3]
        self.label.config(text="   ".join(f"{d}: {probs[d] * 100:.0f}%" for d in top))

    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (280, 280), 0)
        self.pen = ImageDraw.Draw(self.img)
        self.label.config(text="Draw a digit")

App().root.mainloop()