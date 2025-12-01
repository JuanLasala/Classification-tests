import os
import json
from ultralytics import YOLO
from utils import load_config, ensure_dir, log
from plots import plot_loss_curve, plot_confusion, save_classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

# Augmentación moderada para clasificación
AUG_PARAMS = {
    "degrees": 10.0,    # rotación suave (≈ 5–15)
    "scale": 0.25,      # escala moderada (≈ 0.2–0.3)
    "hsv_h": 0.015,     # mantener cerca de los defaults
    "hsv_s": 0.5,
    "hsv_v": 0.4,
}



def train_model():
    # Load config
    config = load_config()
    dataset_path = config["output_dir"]
    class_names = config["classes"]
    seed = config["seed"]

    # Read training parameters from config (single source of truth)
    # Use indexed access to force an obvious error if the keys are missing
    try:
        training_cfg = config["training"]
        epochs = training_cfg["epochs"]
        batch_size = training_cfg["batch_size"]
        img_size = training_cfg["img_size"]
        patience = training_cfg["patience"]
        model_name = training_cfg["model"]
        fraction = training_cfg.get("fraction", 1.0)  # Default to 1.0 if not present
    except KeyError as e:
        missing = e.args[0]
        raise KeyError(
            "Falta la clave '{}' en 'config.json'. Asegúrate de incluir 'training' con: "
            "'epochs', 'batch_size', 'img_size', 'model', 'patience'.".format(missing)
        )
        

    # Output folders
    runs_dir = os.path.join(dataset_path, "..", "runs")
    metrics_dir = os.path.join(runs_dir, "metrics")
    weights_dir = os.path.join(runs_dir, "weights")

    ensure_dir(runs_dir)
    ensure_dir(metrics_dir)
    ensure_dir(weights_dir)

    log(f"Loading YOLOv8 classification model: {model_name} ...")
    model = YOLO(model_name)

    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=patience,
        seed=seed,
        project=runs_dir,
        name="cls_model",
        verbose=True,
        fraction=fraction,
        **AUG_PARAMS,      
    )
    # Save loss plot
    plot_loss_curve(results, metrics_dir)

    # Evaluate on test set
    log("Evaluating model...")
    test_results = model.val(split="test")

    # Extract predictions for confusion matrix
    probs = test_results.probs  # list of arrays
    y_pred = np.array([p.argmax() for p in probs])
    y_true = np.array(test_results.dataset.labels)

    # Save confusion matrix and report
    plot_confusion(y_true, y_pred, class_names, metrics_dir)
    save_classification_report(y_true, y_pred, class_names, metrics_dir)

    # Save weights
    best_model_path = model.export(export_dir=weights_dir)
    log(f"Best model exported to: {best_model_path}")


if __name__ == "__main__":
    train_model()
