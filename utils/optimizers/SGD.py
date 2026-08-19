"""SGD (with momentum) optimizer — an alternative to `Adam` for training
the models in this repo, including the Transformer, through the same
`step()`/`zero_grad()` interface over `model.parameters()`.
"""
import math

class SGD:
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [0 for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            v = self.momentum * self.velocity[i] + self.lr * grad
            self.velocity[i] = v
            p.data = p.data - v

    def zero_grad(self):
        for p in self.params:
            p.grad = None
