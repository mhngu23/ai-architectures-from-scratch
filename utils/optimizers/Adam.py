import math
import numpy as np

class Adam:
    """
    Adam optimizer.
    Args:
        params (list): List of parameters to optimize.
        lr (float): Learning rate.
        betas (tuple): Coefficients used for computing running averages (beta1, beta2).
        eps (float): Term added to denominator to improve numerical stability.
        weight_decay (float, optional): L2 penalty.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # Initialize moments
        self.m = [0 for _ in self.params]
        self.v = [0 for _ in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            # weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            # update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)
            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # update parameters
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None