"""
ORTAK EĞİTİM SCRIPTİ (GENİŞLETİLMİŞ) - train_unified.py
========================================================
Tüm deneyler TEK kod ve TEK adil protokolle. Bu sürüm iki yeni eksen ekler:

  1) ÇOKLU OMURGA (backbone karşılaştırması):
       --backbone {resnet50, vgg19, densenet121, efficientnet_b0}
  2) ABLASYON (omurga donuk mu, ince-ayarlı mı):
       --finetune {none, last_block}
         none       -> omurga TAMAMEN donuk, sadece sınıflandırıcı kafa eğitilir (varsayılan)
         last_block -> omurganın SON bloğu da çözülür (ince-ayar) -> ABLASYON karşılaştırması

Kullanım örnekleri:
    # Ana çok-omurga karşılaştırması (donuk omurga, adil protokol):
    python train_unified.py --dataset cedar --scenario wd --backbone resnet50
    python train_unified.py --dataset cedar --scenario wd --backbone vgg19
    python train_unified.py --dataset cedar --scenario wd --backbone densenet121
    python train_unified.py --dataset cedar --scenario wd --backbone efficientnet_b0

    # Ablasyon (aynı omurga, donuk vs ince-ayar):
    python train_unified.py --dataset cedar --scenario wd --backbone resnet50 --finetune none
    python train_unified.py --dataset cedar --scenario wd --backbone resnet50 --finetune last_block

ADİL PROTOKOL (hepsinde sabit):
    - ImageNet ön-eğitimli omurga
    - 30 epoch + early stopping (patience=5)
    - Adam lr=1e-4, weight_decay=5e-4, batch=32, dropout=0.3
    - 224x224, ImageNet normalizasyonu, seed=42
    (İnce-ayar modunda çözülen katmanlar için lr=1e-5 kullanılır; aşırı bozulmayı önler.)

Çıktılar: outputs_unified/<dataset>_<scenario>_<backbone>_<finetune>/
    best_model.pth, training_history.json, training_curves.png,
    confusion_matrix.png, roc_curve.png, test_metrics.json
"""

import os
import json
import time
import copy
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)

import signature_data as sigdata


# =========================================================
# ADİL PROTOKOL (tüm deneylerde sabit)
# =========================================================
class P:
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5
    LEARNING_RATE = 1e-4          # sınıflandırıcı kafa için
    FINETUNE_LR = 1e-5            # ince-ayarda çözülen omurga katmanları için (daha küçük)
    WEIGHT_DECAY = 5e-4
    DROPOUT = 0.3
    NUM_WORKERS = 0
    SEED = 42
    TEST_RATIO = 0.2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Desteklenen omurgalar ve ImageNet ağırlıkları
BACKBONES = ["resnet50", "vgg19", "densenet121", "efficientnet_b0"]


class ListDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        if len(self.samples) == 0:
            raise RuntimeError("Sample listesi boş.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(img_size, scenario):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # WI'da biraz daha güçlü augmentation; WD'de hafif. (Donmuş omurga, ikisi de ölçülü.)
    if scenario == "wi":
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


def make_head(in_features, num_classes, dropout):
    """Tüm omurgalar için ORTAK sınıflandırıcı kafa (adil protokol)."""
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )


def build_model(backbone, num_classes=2, dropout=0.3, finetune="none"):
    """
    Seçilen omurgayı ImageNet ağırlıklarıyla kurar, omurgayı dondurur,
    ortak sınıflandırıcı kafayı ekler. finetune='last_block' ise omurganın
    son bloğunu yeniden eğitilebilir yapar (ABLASYON).

    Döndürür: (model, finetune_param_list)
      finetune_param_list -> ince-ayarda çözülen omurga parametreleri (ayrı lr için).
    """
    finetune_params = []

    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = make_head(in_features, num_classes, dropout)
        if finetune == "last_block":
            # ResNet'in son evrişim bloğu: layer4
            for p in model.layer4.parameters():
                p.requires_grad = True
                finetune_params.append(p)

    elif backbone == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.classifier[0].in_features   # 25088
        model.classifier = make_head(in_features, num_classes, dropout)
        if finetune == "last_block":
            # VGG'nin son evrişim bloğu: features katmanlarının son ~5'i (block5)
            feat_layers = list(model.features.children())
            for layer in feat_layers[-5:]:
                for p in layer.parameters():
                    p.requires_grad = True
                    finetune_params.append(p)

    elif backbone == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.classifier.in_features      # 1024
        model.classifier = make_head(in_features, num_classes, dropout)
        if finetune == "last_block":
            # DenseNet'in son bloğu: features.denseblock4 (+ norm5)
            for p in model.features.denseblock4.parameters():
                p.requires_grad = True
                finetune_params.append(p)
            for p in model.features.norm5.parameters():
                p.requires_grad = True
                finetune_params.append(p)

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.classifier[1].in_features   # 1280
        model.classifier = make_head(in_features, num_classes, dropout)
        if finetune == "last_block":
            # EfficientNet'in son özellik bloğu: features[-1] (son MBConv + üst conv)
            for p in model.features[-1].parameters():
                p.requires_grad = True
                finetune_params.append(p)

    else:
        raise ValueError(f"Bilinmeyen backbone: {backbone}")

    return model, finetune_params


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
    return (running_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def plot_curves(history, path, title):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    axes[0].set_title(f"Loss ({title})"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc")
    axes[1].set_title(f"Accuracy ({title})"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def plot_cm(y_true, y_pred, class_names, path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.xlabel("Tahmin"); plt.ylabel("Gerçek"); plt.title(f"Confusion Matrix ({title})")
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def plot_roc(y_true, y_probs, path, title):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, "b-", label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({title})"); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cedar", "bhsig", "utsig"])
    ap.add_argument("--scenario", required=True, choices=["wd", "wi"])
    ap.add_argument("--backbone", default="resnet50", choices=BACKBONES,
                    help="Omurga mimarisi (varsayılan: resnet50)")
    ap.add_argument("--finetune", default="none", choices=["none", "last_block"],
                    help="none=omurga donuk (varsayılan); last_block=son blok ince-ayar (ablasyon)")
    args = ap.parse_args()

    torch.manual_seed(P.SEED)
    np.random.seed(P.SEED)

    tag = f"{args.dataset}_{args.scenario}_{args.backbone}_{args.finetune}"
    out_dir = os.path.join("outputs_unified", tag)
    os.makedirs(out_dir, exist_ok=True)
    class_names = ["forged", "genuine"]

    print("=" * 70)
    print(f"DENEY: {args.dataset.upper()} / {args.scenario.upper()} / "
          f"{args.backbone} / finetune={args.finetune}")
    print("=" * 70)
    print(f"Cihaz: {P.DEVICE}")

    # ---- Veri ----
    split = sigdata.make_split(args.dataset, args.scenario,
                               test_ratio=P.TEST_RATIO, seed=P.SEED)
    print(split["info"])
    with open(os.path.join(out_dir, "split_info.txt"), "w", encoding="utf-8") as f:
        f.write(split["info"] + "\n")

    train_tf, val_tf = get_transforms(P.IMG_SIZE, args.scenario)
    train_ds = ListDataset(split["train_samples"], transform=train_tf)
    test_ds = ListDataset(split["test_samples"], transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=P.BATCH_SIZE, shuffle=True,
                              num_workers=P.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=P.BATCH_SIZE, shuffle=False,
                             num_workers=P.NUM_WORKERS, pin_memory=True)

    # ---- Model ----
    model, finetune_params = build_model(args.backbone, num_classes=2,
                                         dropout=P.DROPOUT, finetune=args.finetune)
    model = model.to(P.DEVICE)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"Omurga: {args.backbone} | finetune={args.finetune}")
    print(f"Eğitilebilir parametre: {n_tr:,} / {n_all:,} "
          f"({100.0*n_tr/n_all:.2f}%)")

    criterion = nn.CrossEntropyLoss()

    # Sınıflandırıcı kafa parametreleri (her zaman eğitilir)
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not any(p is fp for fp in finetune_params)]
    param_groups = [{"params": head_params, "lr": P.LEARNING_RATE}]
    if finetune_params:
        # ince-ayarda çözülen omurga katmanları daha küçük lr ile
        param_groups.append({"params": finetune_params, "lr": P.FINETUNE_LR})

    optimizer = optim.Adam(param_groups, weight_decay=P.WEIGHT_DECAY)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_weights, no_improve = 0.0, copy.deepcopy(model.state_dict()), 0
    start = time.time()

    for epoch in range(P.NUM_EPOCHS):
        ep = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, P.DEVICE)
        val_loss, val_acc, _, _, _ = evaluate(model, test_loader, criterion, P.DEVICE)
        scheduler.step(val_loss)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d}/{P.NUM_EPOCHS} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"lr={lr_now:.6f} | süre={time.time()-ep:.1f}s")
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({"model_state_dict": best_weights, "class_names": class_names,
                        "img_size": P.IMG_SIZE, "val_acc": best_acc, "dropout": P.DROPOUT,
                        "backbone": args.backbone, "finetune": args.finetune},
                       os.path.join(out_dir, "best_model.pth"))
            print(f"  -> Yeni en iyi model (val_acc={best_acc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= P.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping! Son {P.EARLY_STOPPING_PATIENCE} epoch'ta iyileşme yok.")
            break

    total_min = (time.time() - start) / 60
    print(f"\nToplam süre: {total_min:.1f} dk | En iyi val_acc: {best_acc:.4f}")

    model.load_state_dict(best_weights)
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ---- Değerlendirme ----
    title = f"{args.dataset.upper()} {args.scenario.upper()} {args.backbone} ft={args.finetune}"
    _, val_acc, preds, labels, probs = evaluate(model, test_loader, criterion, P.DEVICE)
    print("\n" + "=" * 70)
    print(f"TEST DEĞERLENDİRMESİ ({title})")
    print("=" * 70)
    print(f"Test Accuracy:  {val_acc:.4f}")
    print(f"Precision:      {precision_score(labels, preds):.4f}")
    print(f"Recall:         {recall_score(labels, preds):.4f}")
    print(f"F1-Score:       {f1_score(labels, preds):.4f}")
    print("\n" + classification_report(labels, preds, target_names=class_names, digits=4))

    plot_curves(history, os.path.join(out_dir, "training_curves.png"), title)
    plot_cm(labels, preds, class_names, os.path.join(out_dir, "confusion_matrix.png"), title)
    auc = plot_roc(labels, probs, os.path.join(out_dir, "roc_curve.png"), title)
    print(f"ROC AUC:        {auc:.4f}")

    metrics = {
        "dataset": args.dataset, "scenario": args.scenario,
        "backbone": args.backbone, "finetune": args.finetune,
        "trainable_params": int(n_tr), "total_params": int(n_all),
        "epochs_run": len(history["train_loss"]),
        "test_accuracy": float(val_acc),
        "precision": float(precision_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
        "f1_score": float(f1_score(labels, preds)),
        "roc_auc": float(auc),
    }
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nÇıktılar: {out_dir}/")


if __name__ == "__main__":
    main()
