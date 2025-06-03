import numpy as np

class ReLU:
    def __call__(self, x):
        self.last_input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.last_input > 0)