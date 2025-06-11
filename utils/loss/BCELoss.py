import numpy as np

class BCELoss:
    def __call__(self, pred, target):
        self.pred = pred
        self.target = target
        eps = 1e-8  # to avoid log(0)
        return -np.mean(target * np.log(pred + eps) + (1 - target) * np.log(1 - pred + eps))

    def backward(self):
        eps = 1e-8
        return (self.pred - self.target) / ((self.pred + eps) * (1 - self.pred + eps)) / self.pred.size