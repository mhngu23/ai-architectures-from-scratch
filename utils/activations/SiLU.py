"""SiLU (Swish) activation: f(x) = x * sigmoid(x). Smoother than ReLU with
a non-zero gradient for negative inputs; used between the Linear layers of
`DenoiserMLP` (models/diffusion/model.py), matching the activation choice
used by MLP/U-Net denoisers in the DDPM literature.
"""
import numpy as np

class SiLU:
    def __call__(self, x):
        self.last_input = x
        self.last_sigmoid = 1 / (1 + np.exp(-x))
        return x * self.last_sigmoid

    def backward(self, grad_output):
        s = self.last_sigmoid
        x = self.last_input
        local_grad = s + x * s * (1 - s)
        return grad_output * local_grad
