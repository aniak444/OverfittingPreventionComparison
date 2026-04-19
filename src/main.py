import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

from data_loader import DatasetInfo, load_dataset, prepare_and_split_data
from trainer import TrainingConfig, train_model
from metrics import compute_metrics
from overfitting_detector import detect_overfitting

from plots import plot_learning_curves

# quick test
DATASETS = [
    DatasetInfo(name = "Wine", uci_id = 109, columns_to_drop = []),
]

METHODS = [
    TrainingConfig(name = "baseline", hidden_layers = [512, 256, 128]),
    TrainingConfig(name = "dropout",  hidden_layers = [512, 256, 128], dropout_rate = 0.35),
]

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_experiment():
    dataset_info = DATASETS[0]
    print(f"\nRun experiment on dataset: {dataset_info.name}")
    
    features, labels = load_dataset(dataset_info)
    data = prepare_and_split_data(features, labels)

    results_summary = []
    
    for config in METHODS:
        print(f"\nMethod: {config.name}")
        
        result = train_model(data, config)
        
        m_train = compute_metrics(result.model, data.X_train, data.y_train, data.num_classes)
        m_val = compute_metrics(result.model, data.X_val, data.y_val, data.num_classes)
        
        analysis = detect_overfitting(m_train['accuracy'], m_val['accuracy'], m_val['accuracy'], history_dict=result.history.history)
        
        print(f"Train Acc: {m_train['accuracy']:.4f} \nVal Acc: {m_val['accuracy']:.4f}")
        print(f"Status: {analysis.severity.upper()} (Gap: {analysis.train_val_gap:.4f})")


        #plot test
        plot_learning_curves(history = result.history.history, dataset_name = dataset_info.name, method_name = config.name, divergence_epoch = analysis.divergence_epoch,  output_dir = OUTPUT_DIR)
        
        results_summary.append({
            "method": config.name,
            "gap": analysis.train_val_gap,
            "severity": analysis.severity
        })

    print("------------------------")
    print("Test:")
    for r in results_summary:
        print(f" - {r['method']}: Gap={r['gap']:.4f} ({r['severity']})")


if __name__ == "__main__":
    run_experiment()