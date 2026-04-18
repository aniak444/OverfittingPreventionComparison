import time
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from model_builder import build_mlp, count_parameters


EPOCHS = 250
BATCH_SIZE = 16


class TrainingConfig:
    def __init__(self, name, hidden_layers, dropout_rate = 0.0, l1 = 0.0, l2 = 0.0,
                 early_stopping = False, augmentation = False, learning_rate = 1e-3):
        self.name = name
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.l1 = l1
        self.l2 = l2
        self.early_stopping = early_stopping
        self.augmentation = augmentation
        self.learning_rate = learning_rate


class TrainingResult:
    def __init__(self, model, history, config_name, elapsed_sec, num_params):
        self.model = model
        self.history = history
        self.config_name = config_name
        self.elapsed_sec = elapsed_sec
        self.num_params = num_params


def _augment_with_noise(X, y, noise_std=0.05, copies=1):
    X_parts = [X]
    y_parts = [y]
    for _ in range(copies):
        noise = np.random.normal(0.0, noise_std, size=X.shape).astype(np.float32)
        X_parts.append(X + noise)
        y_parts.append(y)
    return np.vstack(X_parts).astype(np.float32), np.concatenate(y_parts).astype(np.int32)


def train_model(prepared_data, config):
    X_train = prepared_data.X_train.copy()
    y_train = prepared_data.y_train.copy()

    if config.augmentation:
        X_train, y_train = _augment_with_noise(X_train, y_train, noise_std=0.05, copies=1)

    model = build_mlp(
        num_features = prepared_data.num_features,
        num_classes = prepared_data.num_classes,
        hidden_layers = config.hidden_layers,
        dropout_rate = config.dropout_rate,
        l1 = config.l1,
        l2 = config.l2,
        learning_rate = config.learning_rate,
    )

    callbacks = []
    if config.early_stopping:
        callbacks.append(
            EarlyStopping(monitor = "val_loss", patience = 20, restore_best_weights = True, verbose = 0)
        )

    start = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data = (prepared_data.X_val, prepared_data.y_val),
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        callbacks = callbacks,
        shuffle = True,
        verbose = 0,
    )
    elapsed = time.perf_counter() - start

    return TrainingResult(
        model = model,
        history = history,
        config_name = config.name,
        elapsed_sec = round(elapsed, 4),
        num_params = count_parameters(model),
    )