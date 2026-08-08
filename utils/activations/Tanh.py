"""Tanh activation: f(x) = tanh(x), giá trị trong (-1, 1). Không được dùng
trực tiếp trong Transformer ở repo này (Transformer dùng ReLU cho FFN và
Sigmoid cho head phân loại) — có sẵn như một activation tổng quát cho các
model khác (vd MLP) trong repo.
"""
import numpy as np

class Tanh:
    def __call__(self, x):
        self.last_input = x
        self.last_output = np.tanh(x)
        return self.last_output

    def backward(self, grad_output):
        return grad_output * (1 - self.last_output ** 2)
