GAP_MODERATE = 0.05
GAP_SEVERE = 0.12


class OverfittingResult:
    def __init__(self, train_val_gap, train_test_gap, severity, divergence_epoch):
        self.train_val_gap = train_val_gap
        self.train_test_gap = train_test_gap
        self.severity = severity
        self.divergence_epoch = divergence_epoch



def get_divergence_epoch(history_dict):
    if not history_dict or 'val_loss' not in history_dict:
        return None
    val_loss = history_dict['val_loss']
    
    min_loss = min(val_loss)
    epoch_number = val_loss.index(min_loss)
    
    return epoch_number + 1



def detect_overfitting(train_acc, val_acc, test_acc, history_dict=None):
    train_val_gap = train_acc - val_acc
    train_test_gap = train_acc - test_acc

    if train_val_gap > GAP_SEVERE or train_test_gap > GAP_SEVERE * 1.2:
        severity = "severe"
    elif train_val_gap > GAP_MODERATE or train_test_gap > GAP_MODERATE * 1.2:
        severity = "moderate"
    else:
        severity = "none"

    divergence_epoch = get_divergence_epoch(history_dict)

    return OverfittingResult(
        train_val_gap = train_val_gap,
        train_test_gap = train_test_gap,
        severity = severity,
        divergence_epoch = divergence_epoch,
    )

