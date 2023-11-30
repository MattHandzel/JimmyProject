class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, val_data_min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.val_data_min_delta = val_data_min_delta

    def early_stop(self, validation_loss):
        if validation_loss < (self.min_validation_loss - self.val_data_min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
