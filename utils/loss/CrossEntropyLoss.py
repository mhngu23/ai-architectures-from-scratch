"""Cross-Entropy Loss — loss function for multi-class classification, used
to train `Seq2SeqTransformer` to predict the next token over the target
vocabulary (as opposed to `BCELoss`, which only handles 2 classes).
Expects `pred` to be probabilities that have already gone through
`Softmax` (not raw logits) — the same activation/loss decoupling as
`Sigmoid` + `BCELoss`. `target` is the correct class index (not one-hot).

Supports `ignore_index` to skip `<pad>` token positions in the target
when training a translator (sentences in a batch have different lengths,
so they must be padded).
"""
import numpy as np


class CrossEntropyLoss:
    def __call__(self, pred, target, ignore_index=None):
        return self.forward(pred, target, ignore_index)

    def forward(self, pred, target, ignore_index=None):
        self.pred = pred
        self.target = target
        eps = 1e-8

        flat_pred = pred.reshape(-1, pred.shape[-1])
        flat_target = target.reshape(-1)
        idx = np.arange(flat_pred.shape[0])

        if ignore_index is not None:
            mask = flat_target != ignore_index
        else:
            mask = np.ones_like(flat_target, dtype=bool)

        self.mask = mask
        self.count = max(mask.sum(), 1)

        correct_probs = flat_pred[idx, flat_target]
        losses = np.where(mask, -np.log(correct_probs + eps), 0.0)
        return losses.sum() / self.count

    def backward(self):
        eps = 1e-8
        flat_pred = self.pred.reshape(-1, self.pred.shape[-1])
        flat_target = self.target.reshape(-1)
        idx = np.arange(flat_pred.shape[0])

        correct_probs = flat_pred[idx, flat_target]
        grad = np.zeros_like(flat_pred)
        grad[idx, flat_target] = np.where(
            self.mask, -1.0 / (correct_probs + eps) / self.count, 0.0
        )
        return grad.reshape(self.pred.shape)
