"""Softmax activation: normalizes logits into a probability distribution
along the last axis. Used as the output activation for multi-class
classification (e.g. `Seq2SeqTransformer` predicting the next token over
the whole target vocabulary), paired with `CrossEntropyLoss` — the same
way `Sigmoid` pairs with `BCELoss` for binary classification.
"""
import numpy as np


class Softmax:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        self.last_output = exp / exp.sum(axis=-1, keepdims=True)
        return self.last_output

    def backward(self, grad_output):
        sum_term = (grad_output * self.last_output).sum(axis=-1, keepdims=True)
        return self.last_output * (grad_output - sum_term)
