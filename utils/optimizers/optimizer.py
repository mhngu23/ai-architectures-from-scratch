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
            p.data = p.data - self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

# Example training loop usage
if __name__ == "__main__":
    # Suppose `model` is an object with a `parameters()` method and a `forward(x)` method,
    # and `loss_fn` computes loss and populates gradients on backward().
    from utils.layers import Linear
    from utils.loss import MSE

    # Dummy model
    class DummyModel:
        def __init__(self):
            self.linear = Linear(1, 1)
        def parameters(self):
            return [self.linear.weight, self.linear.bias]
        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.01)

    # toy data
    data = [(i, 2*i + 1) for i in range(100)]

    for epoch in range(10):
        optimizer.zero_grad()
        total_loss = 0
        for x_val, y_val in data:
            pred = model.forward(x_val)
            loss = loss_fn(pred, y_val)
            loss.backward()
            total_loss += loss.data
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(data):.4f}")