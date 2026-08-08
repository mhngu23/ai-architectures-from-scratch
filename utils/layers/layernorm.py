"""LayerNorm — chuẩn hoá vector đặc trưng của từng token theo trục cuối.
Được dùng sau MỖI residual connection trong EncoderLayer/DecoderLayer
(post-norm: LayerNorm(x + Sublayer(x))) để ổn định phân phối giá trị giữa
các sublayer của Transformer.
"""
import numpy as np

from utils.layers.linear import Parameter


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = Parameter(np.ones((1, dim)))
        self.beta = Parameter(np.zeros((1, dim)))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.last_input = x
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mean) * self.std_inv
        return self.gamma.data * self.xhat + self.beta.data

    def backward(self, grad_output):
        N = grad_output.shape[-1]
        sum_axes = tuple(range(grad_output.ndim - 1))

        grad_gamma = (grad_output * self.xhat).sum(axis=sum_axes).reshape(self.gamma.data.shape)
        grad_beta = grad_output.sum(axis=sum_axes).reshape(self.beta.data.shape)
        self.gamma.grad = grad_gamma if self.gamma.grad is None else self.gamma.grad + grad_gamma
        self.beta.grad = grad_beta if self.beta.grad is None else self.beta.grad + grad_beta

        dxhat = grad_output * self.gamma.data
        sum_dxhat = dxhat.sum(axis=-1, keepdims=True)
        sum_dxhat_xhat = (dxhat * self.xhat).sum(axis=-1, keepdims=True)
        grad_input = self.std_inv / N * (N * dxhat - sum_dxhat - self.xhat * sum_dxhat_xhat)
        return grad_input

    def parameters(self):
        return [self.gamma, self.beta]
