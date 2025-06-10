import math

class SGD:
    """
    Stochastic Gradient Descent optimizer.
    Args:
        params (list): List of parameters to optimize. Each parameter should have .data and .grad attributes.
        lr (float): Learning rate.
        momentum (float, optional): Momentum factor (default: 0.0).
        weight_decay (float, optional): L2 penalty (default: 0.0).
    """
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Initialize velocity for momentum
        self.velocity = [0 for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            # apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            # momentum update
            v = self.momentum * self.velocity[i] + self.lr * grad
            self.velocity[i] = v
            p.data = p.data - v

    def zero_grad(self):
        for p in self.params:
            p.grad = None