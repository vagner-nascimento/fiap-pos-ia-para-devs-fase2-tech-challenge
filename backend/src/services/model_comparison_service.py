"""Service for comparing model predictions and generating metrics/plots."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# Paths
ARTIFACTS_DIR = Path("models/artifacts")
REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"

# Model file names
MODEL_FILES = {
    "best_model": "best_model_predictions.csv",
    "knn_original": "original_knn_predictions.csv",
    "rf_original": "original_rf_predictions.csv",
}


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    REPORTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def load_predictions() -> dict[str, pd.DataFrame]:
    """Load prediction CSVs from artifacts directory."""
    predictions = {}
    for model_name, filename in MODEL_FILES.items():
        file_path = ARTIFACTS_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {file_path}")
        
        logger.info(f"Loading predictions from {file_path}")
        df = pd.read_csv(file_path)
        predictions[model_name] = df
    
    return predictions


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    """Calculate classification metrics for a model."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    return metrics


def generate_confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    class_names: list[str] | None = None,
) -> str:
    """Generate confusion matrix plot and return the URL path."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved confusion matrix plot to {plot_path}")
    # Return URL path relative to backend
    return f"/reports/plots/confusion_matrix_{model_name}.png"


def generate_metrics_comparison_plot(metrics_dict: dict[str, dict[str, float]]) -> str:
    """Generate bar chart comparing metrics across models and return the URL path."""
    # Prepare data for plotting
    models = list(metrics_dict.keys())
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    
    data = []
    for model in models:
        for metric in metric_names:
            data.append({
                "Model": model.replace("_", " ").title(),
                "Metric": metric.replace("_", " ").title(),
                "Value": metrics_dict[model][metric],
            })
    
    df_plot = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x="Metric", y="Value", hue="Model")
    plt.title("Model Metrics Comparison")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / "metrics_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved metrics comparison plot to {plot_path}")
    # Return URL path relative to backend
    return "/reports/plots/metrics_comparison.png"


def generate_class_distribution_plot(y_true: pd.Series, predictions_dict: dict[str, pd.Series]) -> str:
    """Generate plot showing class distribution across models and return the URL path."""
    # Get unique classes
    classes = sorted(y_true.unique())
    
    # Count occurrences for each model
    data = []
    for model_name, y_pred in predictions_dict.items():
        for cls in classes:
            count = (y_pred == cls).sum()
            data.append({
                "Model": model_name.replace("_", " ").title(),
                "Class": str(cls),
                "Count": count,
            })
    
    df_plot = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x="Class", y="Count", hue="Model")
    plt.title("Class Distribution Comparison")
    plt.ylabel("Count")
    plt.xlabel("Class")
    plt.legend(title="Model")
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / "class_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved class distribution plot to {plot_path}")
    # Return URL path relative to backend
    return "/reports/plots/class_distribution.png"


def compare_models() -> dict[str, Any]:
    """
    Compare model predictions and generate report.
    
    Returns:
        Dictionary containing metrics and plot paths.
    """
    ensure_directories()
    
    # Load predictions
    predictions = load_predictions()
    
    # Get class names from the first dataframe
    first_df = list(predictions.values())[0]
    class_names = sorted(first_df["ESTADO_NUTRI"].unique())
    
    # Calculate metrics for each model
    metrics_dict = {}
    predictions_dict = {}
    
    for model_name, df in predictions.items():
        y_true = df["ESTADO_NUTRI"]
        y_pred = df["Prediction"]
        
        metrics = calculate_metrics(y_true, y_pred)
        metrics_dict[model_name] = metrics
        predictions_dict[model_name] = y_pred
        
        logger.info(f"Calculated metrics for {model_name}: {metrics}")
    
    # Generate plots
    first_y_true = list(predictions.values())[0]["ESTADO_NUTRI"]
    
    confusion_plots = {}
    for model_name, df in predictions.items():
        y_pred = df["Prediction"]
        plot_path = generate_confusion_matrix_plot(
            first_y_true,
            y_pred,
            model_name,
            class_names,
        )
        confusion_plots[model_name] = plot_path
    
    metrics_comparison_plot = generate_metrics_comparison_plot(metrics_dict)
    class_distribution_plot = generate_class_distribution_plot(first_y_true, predictions_dict)
    
    # Prepare report
    report = {
        "metrics": metrics_dict,
        "plots": {
            "metrics_comparison": metrics_comparison_plot,
            "class_distribution": class_distribution_plot,
            "confusion_matrices": confusion_plots,
        },
        "class_names": class_names,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    
    # Save report
    report_path = REPORTS_DIR / "model_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved comparison report to {report_path}")
    
    return report
