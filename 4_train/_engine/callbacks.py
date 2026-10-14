class ReduceLROnPlateau:
    """Multiply the LR by `factor` after `patience` epochs without an improvement of the
    validation loss (an epoch only counts as improved if the network is not collapsed)."""

    def __init__(self, factor=0.3, patience=5, min_delta=1e-4, min_lr=1e-6):
        self.factor, self.patience, self.min_delta, self.min_lr = factor, patience, min_delta, min_lr
        self.best = float("inf")
        self.wait = 0

    def on_epoch_end(self, optimizer, val_loss, is_healthy=True):
        if val_loss < self.best - self.min_delta and is_healthy:
            self.best, self.wait = val_loss, 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            for g in optimizer.param_groups:
                g["lr"] = max(g["lr"] * self.factor, self.min_lr)
            print(f"  reducing LR to {optimizer.param_groups[0]['lr']:.2e}")
            self.wait = 0


class EarlyStopping:
    """Signal a stop when the validation loss has not improved for `patience` epochs."""

    def __init__(self, patience=12, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best = float("inf")
        self.wait = 0

    def on_epoch_end(self, val_loss) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best, self.wait = val_loss, 0
            return False
        self.wait += 1
        return self.wait >= self.patience
