import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,precision_score,recall_score,roc_auc_score


def predict_labels(model, X, num_classes):
    predictions = model.predict(X, verbose=0)
    if num_classes == 2:
        return (predictions.ravel() >= 0.5).astype(int)
    return np.argmax(predictions, axis = 1).astype(int)


def predict_scores(model, X, num_classes):
    predictions = model.predict(X, verbose = 0)
    if num_classes == 2:
        return predictions.ravel()
    return predictions


def compute_metrics(model, X, y, num_classes):
    y_pred = predict_labels(model, X, num_classes)
    y_score = predict_scores(model, X, num_classes)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average = "macro", zero_division = 0),
        "recall_macro": recall_score(y, y_pred, average = "macro", zero_division = 0),
        "f1_macro": f1_score(y, y_pred, average = "macro", zero_division = 0),
    }

    try:
        if num_classes == 2:
            metrics["auc"] = roc_auc_score(y, y_score)
        else:
            metrics["auc"] = roc_auc_score(y, y_score, multi_class = "ovr", average = "macro")
    except Exception:
        metrics["auc"] = float("nan")

    return metrics


def compute_all_metrics(model, prepared_data):
    num_classes = prepared_data.num_classes
    
    results = {
        "train": compute_metrics(model, prepared_data.X_train, prepared_data.y_train, num_classes),
        "val": compute_metrics(model, prepared_data.X_val, prepared_data.y_val,num_classes),
        "test": compute_metrics(model, prepared_data.X_test, prepared_data.y_test, num_classes)
    }
    
    return results