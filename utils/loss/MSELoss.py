class MSELoss:
    def __call__(self, pred, target):
        self.pred = pred
        self.target = target
        return ((pred - target) ** 2).mean()

    def backward(self):
        return 2 * (self.pred - self.target) / self.pred.size