import os
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_learning_curves(history, dataset_name, method_name, divergence_epoch, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    #loss curve
    ax1.plot(history['loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Val Loss', color='orange')
    if divergence_epoch is not None:
        ax1.axvline(x=divergence_epoch-1, color='red', linestyle='--', label='Overfitting')
        
    ax1.set_title(f'Loss: {dataset_name} ({method_name})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    #accuracy curve
    ax2.plot(history['accuracy'], label='Train Acc', color='blue')
    ax2.plot(history['val_accuracy'], label='Val Acc', color='orange')
    if divergence_epoch is not None:
        ax2.axvline(x=divergence_epoch-1, color='red', linestyle='--', label='Overfitting')
        
    ax2.set_title(f'Accuracy: {dataset_name} ({method_name})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    filename = f"curves_{dataset_name}_{method_name}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()



def plot_confusion_matrix(y_true, y_pred, class_names, dataset_name, method_name, output_dir):
    c_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {dataset_name} ({method_name})')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    plt.tight_layout()
    filename = f"confusion_matrix_{dataset_name}_{method_name}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()



def plot_roc_curve(y_true, y_score, num_classes, dataset_name, method_name, output_dir):
    plt.figure(figsize=(8, 6))

    if num_classes == 2:
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            plt.plot(false_positive_rate, true_positive_rate, label=f'ROC (AUC = {roc_auc:.2f})')
    else:
         y_true_bin = label_binarize(y_true, classes=range(num_classes))
         for i in range(num_classes):
              false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true_bin[:, i], y_score[:, i])
              roc_auc = auc(false_positive_rate, true_positive_rate)
              plt.plot(false_positive_rate, true_positive_rate, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC: {dataset_name} ({method_name})')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    filename = f"roc_{dataset_name}_{method_name}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()