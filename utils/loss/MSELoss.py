"""Mean Squared Error Loss — used for regression/reconstruction tasks (e.g.
an Autoencoder reconstructing its own input). Not used directly by the
Transformer in this repo (the Transformer uses `BCELoss` for binary
classification), available for other models that need a regression loss.
"""
class MSELoss:
    def __call__(self, pred, target):
        self.pred = pred
        self.target = target
        return ((pred - target) ** 2).mean()

    def backward(self):
        return 2 * (self.pred - self.target) / self.pred.size
