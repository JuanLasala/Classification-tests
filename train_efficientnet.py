import json
import torch
import torch.nn as nn
from timm import create_model
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from plots import plot_loss_curve, plot_confusion, save_classification_report
import os
from datetime import datetime


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

def train(config_path="config.json"):
    # ---- TIMESTAMP ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("RUN ID:", timestamp)

    cfg = load_config(config_path)

    # ---- DIRECTORIOS ÚNICOS POR RUN ----
    checkpoint_dir = os.path.join(cfg["output"]["checkpoint_dir"], timestamp)
    plots_dir = os.path.join(cfg["output"]["plots_dir"], timestamp)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ---- CONFIG ----
    img_size = cfg["dataset"]["img_size"]
    batch_size = cfg["dataset"]["batch_size"]
    lr = cfg["efficientnet"]["learning_rate"]
    epochs = cfg["efficientnet"]["epochs"]
    num_classes = cfg["efficientnet"]["num_classes"]
    model_name = cfg["efficientnet"]["model_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = cfg.get("classes", ["Fire", "No Fire"])

    # ---- TRANSFORMS ----
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    # ---- DATA ----
    train_dataset = datasets.ImageFolder(cfg["dataset"]["train_dir"], transform=tf)
    val_dataset   = datasets.ImageFolder(cfg["dataset"]["val_dir"],   transform=tf)
    test_dataset = datasets.ImageFolder(cfg["dataset"]["test_dir"], transform=tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False, persistent_workers=True)

    # ---- MODEL ----
    print("Cargando modelo EfficientNet preentrenado...")
    model = create_model(model_name, pretrained=True)

    # 🔥 CONGELAR TODA LA EFFICIENTNET
    for param in model.parameters():
        param.requires_grad = False

    # 🔥 2) DESCONGELAR SOLO LOS ÚLTIMOS BLOQUES (fine-tuning parcial)
    for name, param in model.named_parameters():
        if "blocks.5" in name or "blocks.6" in name:  # Ajusta según la arquitectura B0
            param.requires_grad = True


    # 🔥 REEMPLAZAR SOLO LA ÚLTIMA CAPA POR UNA ENTRENABLE
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- LOGS ----
    train_losses = []
    val_losses = []
    val_targets = []
    val_preds = []

    # ---- LOOP ----
    for epoch in range(epochs):

        model.train()
        epoch_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

    # ---- EVALUATION ON VAL SET ----
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss_total += loss.item()
            preds = outputs.argmax(dim=1).cpu().tolist()
            val_preds.extend(preds)
            val_targets.extend(labels.cpu().tolist())

    # store average val loss if there was any validation data
    if len(val_loader) > 0:
        avg_val_loss = val_loss_total / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # ---- SAVE MODEL ----
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "efficientnet_b0.pt"))


     # ---- PLOTS ----
    class Results:
        metrics = type("Metrics", (), {})()

    results = Results()
    results.metrics.train_loss = train_losses
    results.metrics.val_loss = val_losses

    plot_loss_curve(results, plots_dir)

    # Only plot confusion matrix / report if we have predictions
    if len(val_preds) == 0 or len(val_targets) == 0:
        print("No hay muestras de validación. Se omitirá la matriz de confusión y el reporte de clasificación.")
    else:
        plot_confusion(val_targets, val_preds, class_names, plots_dir)
        save_classification_report(val_targets, val_preds, class_names, plots_dir)

    print("\nEntrenamiento finalizado.")
    print("Los gráficos están guardados en:", cfg["output"]["plots_dir"])



if __name__ == "__main__":
    train()
