import numpy as np

class Sigmoid:
    """Sigmoid activation function.
    This class implements the sigmoid activation function, which is defined as:
    f(x) = 1 / (1 + exp(-x))
    It also provides a method for backpropagation to compute the gradient of the loss with respect to the input.
    The sigmoid function squashes the input to a range between 0 (small input) and 1 (large input), making it useful for binary classification tasks.
    When the input is very large or very small, the output approaches 1 or 0, respectively, which can lead to vanishing gradients during training.
    Attributes:
        last_input (numpy.ndarray): Stores the last input passed to the activation function for use in backpropagation.
        last_output (numpy.ndarray): Stores the last output of the activation function for use in backpropagation.
    """
    def __call__(self, x):
        self.last_input = x
        self.last_output = 1 / (1 + np.exp(-x))  # Save output for backward
        return self.last_output

    def backward(self, grad_output):
        # The sigmoid function is defined as f(x) = 1 / (1 + exp(-x))
        # Taking the gradient of the sigmoid function 1 / u(x) where u(x) = 1 + exp(-x)
        # 1 /u(x) = u(x)^-1 = -u(x)^-2 * du(x)/dx = -1/(u(x)^2) * (-exp(-x)) = exp(-x) / (1 + exp(-x))^2 or f(x) * (1 - f(x))
        # Note: derivative of e(-x) is -e(-x)
        # The derivative of the sigmoid function is f'(x) = f(x) * (1 - f(x))
        return grad_output * self.last_output * (1 - self.last_output)