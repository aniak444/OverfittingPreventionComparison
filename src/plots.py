import os
import matplotlib.pyplot as plt

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