import numpy as np

class Tanh:
    """Tanh activation function.
    This class implements the hyperbolic tangent activation function, which is defined as:
    f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    It also provides a method for backpropagation to compute the gradient of the loss with respect to the input.
    The tanh function squashes the input to a range between -1 and 1, centering activations around zero,
    which can help with convergence compared to sigmoid.
    Attributes:
        last_input (numpy.ndarray): Stores the last input passed to the activation function for use in backpropagation.
        last_output (numpy.ndarray): Stores the last output of the activation function for use in backpropagation.
    """
    def __call__(self, x):
        self.last_input = x
        self.last_output = np.tanh(x)  # Save output for backward
        return self.last_output

    def backward(self, grad_output):
        # The derivative of tanh is f'(x) = 1 - tanh(x)^2
        return grad_output * (1 - self.last_output ** 2)