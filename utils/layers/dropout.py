"""Dropout — randomly zeroes activations (scaled by 1/(1-p), "inverted
dropout") during training to keep the network from memorizing training
examples instead of learning general structure. A no-op during inference
(`self.training = False`): `Encoder`/`Decoder`/the top-level Transformer
models expose `.train()`/`.eval()` to toggle every `Dropout` inside them,
mirroring `week6_demo.ipynb`'s finding that this repo's Transformer had no
regularization and would overfit tiny datasets.
"""
import numpy as np


class Dropout:
    def __init__(self, p=0.1):
        self.p = p
        self.training = True

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if not self.training or self.p <= 0:
            self.mask = None
            return x
        self.mask = (np.random.rand(*x.shape) >= self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, grad_output):
        if self.mask is None:
            return grad_output
        return grad_output * self.mask

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
