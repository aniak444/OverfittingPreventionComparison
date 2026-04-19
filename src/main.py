import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import pandas as pd

from data_loader import DatasetInfo, load_dataset, prepare_and_split_data
from trainer import TrainingConfig, train_model
from metrics import compute_metrics, predict_labels
from overfitting_detector import detect_overfitting

from plots import plot_learning_curves, plot_confusion_matrix, plot_roc_curve, plot_overfitting_heatmap, plot_accuracy_comparison, plot_training_time_comparison


DATASETS = [
    DatasetInfo(name = "Wine", uci_id = 109, columns_to_drop = []),
    DatasetInfo(name="Breast_Cancer", uci_id=17, columns_to_drop=["ID"]),
    DatasetInfo(name="Iris", uci_id=53, columns_to_drop=[]),
]

METHODS = [
    TrainingConfig(name = "baseline", hidden_layers = [512, 256, 128]),
    TrainingConfig(name = "dropout",  hidden_layers = [512, 256, 128], dropout_rate = 0.35),
    TrainingConfig(name="l1_reg", hidden_layers=[512, 256, 128], l1=0.001),
    TrainingConfig(name="l2_reg", hidden_layers=[512, 256, 128], l2=0.001),
    TrainingConfig(name="early_stopping", hidden_layers=[512, 256, 128], early_stopping=True),
    TrainingConfig(name="simple_model", hidden_layers=[64, 32]),
    TrainingConfig(name="augmentation", hidden_layers=[512, 256, 128], augmentation=True),
    TrainingConfig(name="smote", hidden_layers=[512, 256, 128], smote=True),
]

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_experiment():
    all_results = []
    for dataset_info in DATASETS:
        print("\n")
        print(f"\nDataset: {dataset_info.name}")
        features, labels = load_dataset(dataset_info)
        data = prepare_and_split_data(features, labels)

        print(f"Samples: train: {len(data.X_train)}, val: {len(data.X_val)}, test: {len(data.X_test)}")
        print(f"Features: {data.num_features}\n Classes: {data.num_classes} [{data.category_names}]")

        for config in METHODS:
            print(f"\nMethod: {config.name}")
        
            result = train_model(data, config)

            m_train = compute_metrics(result.model, data.X_train, data.y_train, data.num_classes)
            m_val = compute_metrics(result.model, data.X_val,   data.y_val,   data.num_classes)
            m_test = compute_metrics(result.model, data.X_test,  data.y_test,  data.num_classes)

            analysis = detect_overfitting(m_train['accuracy'], m_val['accuracy'], m_test['accuracy'], history_dict=result.history.history)

            print(f"Accuracy: Train: {m_train['accuracy']:.4f} Val: {m_val['accuracy']:.4f} Test: {m_test['accuracy']:.4f}")
            print(f"Additional: F1: {m_test['f1_macro']:.4f} AUC: {m_test['auc']:.4f}")
            print(f"Status: {analysis.severity.upper()} (gap: {analysis.train_val_gap:.4f})")
            print(f"Time: {result.elapsed_sec:.1f} sec.")

            #plots
            plot_learning_curves(history = result.history.history, dataset_name = dataset_info.name, method_name = config.name, divergence_epoch = analysis.divergence_epoch,  output_dir = OUTPUT_DIR)
        
            y_pred = predict_labels(result.model, data.X_val, data.num_classes)
            plot_confusion_matrix(y_true = data.y_val, y_pred = y_pred, class_names = data.category_names, dataset_name = dataset_info.name, method_name = config.name, output_dir = OUTPUT_DIR)

            y_score = result.model.predict(data.X_val, verbose=0)
            plot_roc_curve(y_true = data.y_val, y_score = y_score, num_classes = data.num_classes, dataset_name = dataset_info.name, method_name = config.name, output_dir = OUTPUT_DIR)

            all_results.append({
                "dataset": dataset_info.name,
                "method": config.name,
                "train_accuracy": round(m_train["accuracy"], 4),
                "val_accuracy": round(m_val["accuracy"], 4),
                "test_accuracy": round(m_test["accuracy"], 4),
                "f1_macro": round(m_test["f1_macro"], 4),
                "auc": round(m_test["auc"], 4),
                "train_val_gap": round(analysis.train_val_gap, 4),
                "train_test_gap": round(analysis.train_test_gap, 4),
                "severity": analysis.severity,
                "divergence_epoch": analysis.divergence_epoch,
                "num_params": result.num_params,
                "train_time_sec": result.elapsed_sec,
            })

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved.")

    plot_overfitting_heatmap(results_df, OUTPUT_DIR)
    plot_accuracy_comparison(results_df, OUTPUT_DIR)
    plot_training_time_comparison(results_df, OUTPUT_DIR)

    print("\nSummary:")
    cols = ["dataset", "method", "test_accuracy", "f1_macro", "auc", "train_val_gap", "severity", "train_time_sec"]
    print(results_df[cols].to_string(index=False))



if __name__ == "__main__":
    run_experiment()