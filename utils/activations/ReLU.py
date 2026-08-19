"""ReLU activation: f(x) = max(0, x). Used as the non-linearity between the
2 Linear layers in `PositionwiseFeedForward` of every Encoder/DecoderLayer
in the Transformer, as well as in MLP/Autoencoder elsewhere in the repo.
"""
import numpy as np

class ReLU:
    def __call__(self, x):
        self.last_input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.last_input > 0)
