"""Linear (fully-connected) layer + Parameter wrapper — the most basic
building block in the whole repo. Every projection in the Transformer
reuses this class: Q/K/V/output projections in MultiHeadAttention, the
2 layers of PositionwiseFeedForward, per-column projection in
FeatureTokenizer, the final logit projection in
TabularTransformer/TextClassifierTransformer/Seq2SeqTransformer — as
well as in MLP and Autoencoder elsewhere in the repo.
"""
import numpy as np

class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = None


class Linear:
    def __init__(self, in_features, out_features, activation=None):
        if activation == 'relu':
            self.W = Parameter(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features))
        elif activation == 'sigmoid' or activation == 'tanh':
            self.W = Parameter(np.random.randn(in_features, out_features) * np.sqrt(1. / in_features))
        else:
            self.W = Parameter(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features))

        self.b = Parameter(np.zeros((1, out_features)))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.last_input = x
        return x @ self.W.data + self.b.data

    def backward(self, grad_output):
        grad_W = self.last_input.T @ grad_output
        grad_b = grad_output.sum(axis=0, keepdims=True)

        self.W.grad = grad_W if self.W.grad is None else self.W.grad + grad_W
        self.b.grad = grad_b if self.b.grad is None else self.b.grad + grad_b

        grad_input = grad_output @ self.W.data.T

        return grad_input
