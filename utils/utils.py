import torch
class EarlyStopping():
    """
    stop the training when the loss does not improve.
    """
    def __init__(self, patience=50, mode='min'):
        if mode not in ['min', 'max']:
            raise ValueError("Early-stopping mode not supported")
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_val = None

    def __call__(self, val):
        val = float(val)
        if self.best_val is None:
            self.best_val = val
        elif self.mode == 'min' and val < self.best_val:
            self.best_val = val
            self.counter = 0
        elif self.mode == 'max' and val > self.best_val:
            self.best_val = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early Stopping!")
                return True
        return False


def check_early_stopping(val_loss, model, early_stopping, epoch, best_model_path):
    if early_stopping.best_val is None or val_loss < early_stopping.best_val:
        early_stopping.best_val = val_loss  # Aggiorna la migliore loss
        early_stopping.counter = 0  # Resetta il contatore
        torch.save(model.state_dict(), best_model_path)  # Salva i pesi migliori
        print(f"Saved best model with val_loss: {val_loss:.4f} at epoch {epoch}")
        return False  # Non fermare il training
    else:
        early_stopping.counter += 1
        if early_stopping.counter >= early_stopping.patience:
            print(f"Early stopping at epoch {epoch} with best val_loss: {early_stopping.best_val:.4f}")
            return True  # Fermare il training
    return False  # Continuare il training