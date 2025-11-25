import os
import json
from ultralytics import YOLO
from utils import load_image_paths, ensure_dir
import csv
import shutil

def predict_batch(config_path="config.json"):
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    model_path = config.get("model_path", "yolov8n-cls.pt")
    input_dir = config["prediction_input_dir"]
    output_dir = config["prediction_output_dir"]
    class_names = config["classes"]

    ensure_dir(output_dir)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Collecting images from: {input_dir}")
    image_paths = load_image_paths(input_dir)

    print(f"Found {len(image_paths)} images to classify.\n")

    results = []
    
    for img in image_paths:
        r = model(img)[0]
        pred_class_idx = r.probs.top1
        pred_class_name = class_names[pred_class_idx]

        # Save to class directory
        class_dir = os.path.join(output_dir, pred_class_name)
        ensure_dir(class_dir)
        shutil.copy(img, os.path.join(class_dir, os.path.basename(img)))

        results.append([img, pred_class_name])

    # Save CSV
    csv_path = os.path.join(output_dir, "predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "predicted_class"])
        writer.writerows(results)

    print(f"Predictions saved to: {csv_path}")
    print(f"Classified images copied into: {output_dir}")

if __name__ == "__main__":
    predict_batch()
