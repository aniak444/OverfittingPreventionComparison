import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


#builds and compiles a mlp model
def build_mlp(num_features, num_classes, hidden_layers, dropout_rate = 0.0, l1 = 0.0, l2 = 0.0, learning_rate = 1e-3):
    reg = regularizers.L1L2(l1 = l1, l2 = l2)

    model = keras.Sequential(name = "mlp")
    model.add(layers.Input(shape = (num_features,)))

    for units in hidden_layers:
        model.add(layers.Dense(units, activation = "relu", kernel_regularizer = reg))
        if dropout_rate > 0.0:
            model.add(layers.Dropout(dropout_rate))

    if num_classes == 2:
        model.add(layers.Dense(1, activation = "sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(layers.Dense(num_classes, activation = "softmax"))
        loss = "sparse_categorical_crossentropy"

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
        loss = loss,
        metrics = ["accuracy"],
    )
    return model


def count_parameters(model):
    return int(model.count_params())