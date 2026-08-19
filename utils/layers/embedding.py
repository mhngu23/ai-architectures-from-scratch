"""Tokenizing layer for TEXT data, used by `TextClassifierTransformer` and
`Seq2SeqTransformer`. `TokenEmbedding` looks up an embedding table by word
index (unlike `FeatureTokenizer`, which is only for tabular numeric data);
`PositionalEncoding` adds fixed sin/cos position information, since
attention by itself is position-agnostic and word order in a sentence
carries semantic meaning — unlike column order in tabular data.
"""
import numpy as np

from utils.layers.linear import Parameter


class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.table = Parameter(np.random.randn(vocab_size, d_model) * 0.02)

    def __call__(self, token_ids):
        return self.forward(token_ids)

    def forward(self, token_ids):
        self.token_ids = token_ids
        return self.table.data[token_ids]

    def backward(self, grad_output):
        grad_table = np.zeros_like(self.table.data)
        np.add.at(grad_table, self.token_ids, grad_output)
        self.table.grad = grad_table if self.table.grad is None else self.table.grad + grad_table

    def parameters(self):
        return [self.table]


class PositionalEncoding:
    def __init__(self, d_model, max_len=512):
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]

    def backward(self, grad_output):
        return grad_output

    def parameters(self):
        return []
