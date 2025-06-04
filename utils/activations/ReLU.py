import numpy as np

class ReLU:
    def __call__(self, x):
        self.last_input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        # Relu function is defined as f(x) = max(0, x) or f(x) = x if x > 0 else 0
        # The gradient of Relu f(x) is 1 if x > 0, and 0 if x <= 0
        # (self.last_input > 0) creates a boolean mask where the condition is true 
        return grad_output * (self.last_input > 0) 