"""ReLU activation: f(x) = max(0, x). Dùng làm phi tuyến tính ở giữa 2 lớp
Linear trong `PositionwiseFeedForward` của mỗi Encoder/DecoderLayer trong
Transformer, cũng như trong MLP/Autoencoder ở các phần khác của repo.
"""
import numpy as np

class ReLU:
    def __call__(self, x):
        self.last_input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.last_input > 0)
