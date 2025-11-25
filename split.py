import os
import json
import random
import shutil
import sys
from pathlib import Path

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)

def show_progress(current, total, label=""):
    """Muestra una barra de progreso simple"""
    percent = (current / total) * 100
    bar_length = 30
    filled = int(bar_length * current // total)
    bar = "█" * filled + "░" * (bar_length - filled)
    sys.stdout.write(f"\r{label} [{bar}] {percent:.1f}% ({current}/{total})")
    sys.stdout.flush()
    if current == total:
        print()  # Salto de línea al terminar

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def split_list(data, train_ratio, val_ratio, test_ratio):
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test

def main():
    # Load configuration
    config = load_config("config.json")

    source_dir = Path(config["source_dir"])
    output_dir = Path(config["output_dir"])
    splits = config["splits"]
    classes = config["classes"]
    seed = config["seed"]

    random.seed(seed)

    print("=== Dataset Splitter ===")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")

    # Create output directory structure
    print("\n⏳ Creando estructura de directorios...")
    for split in ["train", "val", "test"]:
        for cls in classes:
            create_dir(output_dir / split / cls)
    print("✓ Directorios creados")

    for cls in classes:
        class_dir = source_dir / cls
        if not class_dir.exists():
            print(f"❌ ERROR: No se encontró la carpeta: {class_dir}")
            continue

        image_files = [f for f in class_dir.iterdir() if f.is_file()]

        print(f"\n⏳ Procesando clase: {cls}")
        print(f"   Total imágenes: {len(image_files)}")

        # Perform split
        train_files, val_files, test_files = split_list(
            image_files,
            splits["train"],
            splits["val"],
            splits["test"]
        )

        print(f"   ├─ Train: {len(train_files)}")
        print(f"   ├─ Val:   {len(val_files)}")
        print(f"   └─ Test:  {len(test_files)}")

        # Copy files
        print(f"\n   ⏳ Copiando archivos para {cls}...")
        total_files = len(train_files) + len(val_files) + len(test_files)
        file_count = 0
        
        for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            for f in files:
                shutil.copy(f, output_dir / split / cls)
                file_count += 1
                show_progress(file_count, total_files, f"   Copiando")
        print(f"   ✓ {total_files} archivos copiados")

    print("\n✅ DONE! Dataset ready for YOLOv8 classification.")
    print("=" * 50)

if __name__ == "__main__":
    main()
