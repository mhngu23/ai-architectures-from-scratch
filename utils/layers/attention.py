"""Cơ chế Attention — trái tim của kiến trúc Transformer. `ScaledDotProductAttention`
tính công thức attention gốc; `MultiHeadAttention` bọc thêm các phép chiếu
Q/K/V/output học được và chia thành nhiều head song song. Được dùng làm
self-attention (trong Encoder, và masked self-attention trong Decoder) lẫn
cross-attention (Decoder chú ý sang output của Encoder) xuyên suốt
EncoderLayer/DecoderLayer trong `models/transformer/model.py`.
"""
import numpy as np

from utils.layers.linear import Linear


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


class ScaledDotProductAttention:
    def __call__(self, Q, K, V, mask=None):
        return self.forward(Q, K, V, mask)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        self.attn_weights = softmax(scores, axis=-1)
        self.Q, self.K, self.V = Q, K, V
        return self.attn_weights @ V

    def backward(self, grad_output):
        d_k = self.Q.shape[-1]

        grad_attn = grad_output @ np.swapaxes(self.V, -1, -2)
        grad_V = np.swapaxes(self.attn_weights, -1, -2) @ grad_output

        sum_term = (grad_attn * self.attn_weights).sum(axis=-1, keepdims=True)
        grad_scores = self.attn_weights * (grad_attn - sum_term)
        grad_scores = grad_scores / np.sqrt(d_k)

        grad_Q = grad_scores @ self.K
        grad_K = np.swapaxes(grad_scores, -1, -2) @ self.Q
        return grad_Q, grad_K, grad_V


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def __call__(self, Q_in, K_in, V_in, mask=None):
        return self.forward(Q_in, K_in, V_in, mask)

    def _project(self, linear, x):
        batch, seq_len, _ = x.shape
        out = linear(x.reshape(batch * seq_len, -1))
        return out.reshape(batch, seq_len, -1)

    def _project_backward(self, linear, grad, batch, seq_len):
        grad_in = linear.backward(grad.reshape(batch * seq_len, -1))
        return grad_in.reshape(batch, seq_len, -1)

    def _split_heads(self, x):
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        batch, heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq_len, heads * d_k)

    def forward(self, Q_in, K_in, V_in, mask=None):
        self.batch, self.seq_q, _ = Q_in.shape
        self.seq_k = K_in.shape[1]

        Q = self._split_heads(self._project(self.W_q, Q_in))
        K = self._split_heads(self._project(self.W_k, K_in))
        V = self._split_heads(self._project(self.W_v, V_in))

        context = self.attention(Q, K, V, mask)
        context = self._merge_heads(context)
        return self._project(self.W_o, context)

    def backward(self, grad_output):
        grad_context = self._project_backward(self.W_o, grad_output, self.batch, self.seq_q)
        grad_context = self._split_heads(grad_context)

        grad_Q, grad_K, grad_V = self.attention.backward(grad_context)

        grad_Q = self._merge_heads(grad_Q)
        grad_K = self._merge_heads(grad_K)
        grad_V = self._merge_heads(grad_V)

        grad_Q_in = self._project_backward(self.W_q, grad_Q, self.batch, self.seq_q)
        grad_K_in = self._project_backward(self.W_k, grad_K, self.batch, self.seq_k)
        grad_V_in = self._project_backward(self.W_v, grad_V, self.batch, self.seq_k)
        return grad_Q_in, grad_K_in, grad_V_in

    def parameters(self):
        params = []
        for linear in (self.W_q, self.W_k, self.W_v, self.W_o):
            params.append(linear.W)
            params.append(linear.b)
        return params
