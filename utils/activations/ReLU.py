import numpy as np

class ReLU:
    """"Rectified Linear Unit (ReLU) activation function.
    This class implements the ReLU activation function, which is defined as:
    f(x) = max(0, x)
    It also provides a method for backpropagation to compute the gradient of the loss with respect to the input.
    The idea behind ReLU is to allow only positive values to pass through, effectively setting all negative values to zero.
    This helps to mitigate the vanishing gradient problem (does not squash gradient of positive values) and allows models to learn faster and perform better.
    This is because it introduces non-linearity into the model while being computationally efficient (it only contains simple threshold no need to compute exponetial like tanh and sigmoid).
    Since Relu output is zero for negative inputs, it can lead to "dead neurons" during training, where the gradient is zero and the weights do not update. 
    Attributes:
        last_input (numpy.ndarray): Stores the last input passed to the activation function for use in backpropagation.
    """
    def __call__(self, x):
        self.last_input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        # Relu function is defined as f(x) = max(0, x) or f(x) = x if x > 0 else 0
        # The gradient of Relu f(x) is 1 if x > 0, and 0 if x <= 0
        # (self.last_input > 0) creates a boolean mask where the condition is true 
        return grad_output * (self.last_input > 0) 