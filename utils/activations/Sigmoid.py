import numpy as np

class Sigmoid:
    def __call__(self, x):
        self.last_input = x
        self.last_output = 1 / (1 + np.exp(-x))  # Save output for backward
        return self.last_output

    def backward(self, grad_output):
        # The sigmoid function is defined as f(x) = 1 / (1 + exp(-x))
        # Taking the gradient of the sigmoid function 1 / u(x) where u(x) = 1 + exp(-x)
        # 1 /u(x) = u(x)^-1 = -1 * u(x)^-2 * du(x)/dx = -1/(u(x)^2) * (-exp(-x)) = exp(-x) / (1 + exp(-x))^2 or f(x) * (1 - f(x))
        # The derivative of the sigmoid function is f'(x) = f(x) * (1 - f(x))
        return grad_output * self.last_output * (1 - self.last_output)