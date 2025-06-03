import numpy as np

class Linear:
    def __init__(self, in_features, out_features, activation=None):
        """
        Initializes a linear layer with weights and biases.
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        # Initialize weights and biases
        # When initializing the weights, we aim to make the variance of the output similar to the variance of the input.
        # The reason is that if the variance of the output is too high or too low, it can lead to issues like vanishing or exploding gradients during training.
        # This is done by scaling the weight based on the number of input features.
        if activation == 'relu':
            # He initialization is commonly used for ReLU activation functions
            # It initializes the weights with a variance of 2/n, where n is the number of input features.
            # This is because RelU activation can lead to dead neurons (neurons that always output zero as ReLU turns all value smaller than 0 to become 0) if the weights are initialized too small.	
            # This also explains why we use 2./ instead of 1./ like in Xavier initialization.
            self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features) # He initialization:  https://arxiv.org/abs/1502.01852
        elif activation == 'sigmoid' or activation == 'tanh':
            self.W = np.random.randn(in_features, out_features) * np.sqrt(1. / in_features) # Xavier initialization: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        else:
            self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)

        self.b = np.zeros((1, out_features))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.last_input = x
        return x @ self.W + self.b

    def backward(self, grad_output, lr=1e-3):
        """
        Computes the gradient of the loss with respect to the input and updates the weights and biases.
        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output of this layer.
            lr (float): Learning rate for updating weights and biases.
        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer.
        """
        # Compute the gradients of weights as the outer product of the last input and the gradient of the output. 
        # This is because the gradient of the output with respect to the weights (d_L\d_W) is the input to this layer.
        # We use chain rule here: d_L/d_W = d_L/d_y * d_y/d_W, where y = Wx + b.
        grad_W = self.last_input.T @ grad_output
        # Compute the gradients of biases as the sum of the gradients of the output along the batch dimension.
        # This is because the gradient of the output with respect to the biases (d_L\d_b) is the sum of the gradients of the output.
        # We use chain rule here: d_L/d_b = d_L/d_y * d_y/d_b, where y = Wx + b.
        grad_b = grad_output.sum(axis=0, keepdims=True)

        self.W -= lr * grad_W
        self.b -= lr * grad_b

        grad_input = grad_output @ self.W.T

        return grad_input