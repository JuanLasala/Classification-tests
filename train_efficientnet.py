import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
from utils import load_config, set_seed, ensure_dir, load_image_paths
from datasets import ImageClassificationDataset
#from plots import plot_training_curves

def build_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

def build_dataloaders(config):
    train_paths, val_paths = load_image_paths(config["dataset"]["train_dir"], config["dataset"]["val_dir"])

    train_tf = build_transforms(config["training"]["image_size"])
    val_tf   = build_transforms(config["training"]["image_size"])

    train_dataset = ImageClassificationDataset(train_paths, transform=train_tf)
    val_dataset   = ImageClassificationDataset(val_paths,   transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)

    return train_loader, val_loader

def build_model(config):
    model = create_model(
        "efficientnet_b0",
        pretrained=config["model"]["pretrained"],
        num_classes=config["model"]["num_classes"]
    )
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

def train(config_path="config.json"):
    # ---- CONFIG ----
    config = load_config(config_path)
    set_seed(config["training"]["seed"])
    ensure_dir(config["output"]["model_dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- DATA ----
    train_loader, val_loader = build_dataloaders(config)

    # ---- MODEL ----
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    best_val_acc = 0
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}

    # ---- TRAIN LOOP ----
    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[{epoch+1}/{config['training']['epochs']}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(config["output"]["model_dir"], "efficientnet_b0_best.pt"))

    # ---- PLOTTING ----
    if config["output"].get("plot_curves", False):
        plot_training_curves(history)

    print("Entrenamiento finalizado. Mejor accuracy:", best_val_acc)
    

if __name__ == "__main__":
    train()
