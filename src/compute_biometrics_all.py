"""
TÜM MODELLER İÇİN BİYOMETRİK METRİKLER - compute_biometrics_all.py
====================================================================
Kaydedilmiş tüm best_model.pth dosyalarını test setinde çalıştırıp
FAR, FRR, AER ve EER hesaplar. Modelleri YENİDEN EĞİTMEZ; sadece
kayıtlı ağırlıkları yükleyip test verisinde olasılık skorlarını toplar.

Çıktı:
    outputs_unified/_biometrics/biometric_all.csv   (tüm modeller tek tabloda)
    Ayrıca her senaryo için ROC tabanlı EER'i ekranda yazar.

Kullanım:
    python compute_biometrics_all.py
    (signature_data.py aynı klasörde olmalı; veri yolları onun içinde ayarlı)

Gereksinim: torch, torchvision, numpy, scikit-learn, pillow
"""

import os
import csv
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score

import signature_data as sigdata

OUTPUT_DIR = "outputs_unified"
BIO_DIR = os.path.join(OUTPUT_DIR, "_biometrics")
os.makedirs(BIO_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
TEST_RATIO = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROPOUT = 0.3


# ---- ListDataset (train scriptindekiyle aynı) ----
class ListDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def make_head(in_features, num_classes, dropout):
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )


def build_model(backbone, num_classes=2, dropout=0.3):
    """Mimariyi kurar (ağırlıklar sonradan state_dict'ten yüklenecek, ImageNet indirmeye gerek yok)."""
    if backbone == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = make_head(model.fc.in_features, num_classes, dropout)
    elif backbone == "vgg19":
        model = models.vgg19(weights=None)
        model.classifier = make_head(model.classifier[0].in_features, num_classes, dropout)
    elif backbone == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = make_head(model.classifier.in_features, num_classes, dropout)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier = make_head(model.classifier[1].in_features, num_classes, dropout)
    else:
        raise ValueError(f"Bilinmeyen backbone: {backbone}")
    return model


def compute_biometrics(probs_genuine, labels):
    """
    probs_genuine: her örnek için 'genuine' (sınıf 1) olasılığı
    labels: gerçek etiketler (1=genuine, 0=forged)
    FAR: sahteyi gerçek kabul etme oranı
    FRR: gerçeği sahte sayma (reddetme) oranı
    EER: FAR=FRR olduğu nokta
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs_genuine)

    # 0.5 eşiğinde FAR/FRR
    pred = (probs >= 0.5).astype(int)
    # forged = 0, genuine = 1
    n_forged = (labels == 0).sum()
    n_genuine = (labels == 1).sum()
    # FAR = sahte(0) iken genuine(1) tahmin edilen / toplam sahte
    far = ((pred == 1) & (labels == 0)).sum() / max(1, n_forged)
    # FRR = gerçek(1) iken forged(0) tahmin edilen / toplam gerçek
    frr = ((pred == 0) & (labels == 1)).sum() / max(1, n_genuine)
    aer = (far + frr) / 2

    # EER: ROC üzerinde FAR(=FPR) ile FRR(=1-TPR=FNR) kesişimi
    # roc_curve genuine'i pozitif sınıf alır
    fpr, tpr, thr = roc_curve(labels, probs, pos_label=1)
    fnr = 1 - tpr
    # FPR ve FNR'nin en yakın olduğu nokta
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2

    auc = roc_auc_score(labels, probs)
    return far, frr, aer, eer, auc


@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # genuine olasılığı
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def parse_folder(name):
    """
    Klasör adından (dataset, scenario, backbone, finetune) çıkarır.
    Biçimler:
      eski:  <dataset>_<scenario>                       (varsayılan resnet50/none)
      yeni:  <dataset>_<scenario>_<backbone>_<finetune>
    """
    parts = name.split("_")
    ds = parts[0]
    sc = parts[1]
    rest = parts[2:]  # backbone (+ belki finetune) parçaları

    if not rest:
        # eski format: <dataset>_<scenario>
        return ds, sc, "resnet50", "none"

    # finetune değeri yalnızca "none" veya "last_block" olabilir.
    # Sondan tanı: "last_block" iki parça (last, block), "none" tek parça.
    tail2 = "_".join(rest[-2:]) if len(rest) >= 2 else ""
    if tail2 == "last_block":
        finetune = "last_block"
        backbone = "_".join(rest[:-2])
    elif rest[-1] == "none":
        finetune = "none"
        backbone = "_".join(rest[:-1])
    else:
        # finetune eki yok (çok eski format), tümü backbone
        finetune = "none"
        backbone = "_".join(rest)

    if not backbone:
        backbone = "resnet50"
    return ds, sc, backbone, finetune


def main():
    if not os.path.isdir(OUTPUT_DIR):
        print(f"'{OUTPUT_DIR}' yok. Önce train_unified.py çalıştırın.")
        return

    tf = val_transform()
    rows = []
    # split'leri yeniden üretmemek için cache (aynı ds_sc için bir kez)
    split_cache = {}

    folders = sorted(d for d in os.listdir(OUTPUT_DIR)
                     if os.path.isdir(os.path.join(OUTPUT_DIR, d))
                     and os.path.isfile(os.path.join(OUTPUT_DIR, d, "best_model.pth")))

    print(f"{len(folders)} model bulundu.\n")

    for name in folders:
        ds, sc, backbone, finetune = parse_folder(name)
        ckpt_path = os.path.join(OUTPUT_DIR, name, "best_model.pth")
        print(f"[{name}] -> ds={ds} sc={sc} backbone={backbone} ft={finetune}")

        # Test split (aynı seed -> train scriptiyle birebir aynı test kümesi)
        key = (ds, sc)
        if key not in split_cache:
            split_cache[key] = sigdata.make_split(ds, sc, test_ratio=TEST_RATIO, seed=SEED)
        split = split_cache[key]
        test_ds = ListDataset(split["test_samples"], transform=tf)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Model kur + ağırlık yükle
        try:
            model = build_model(backbone, num_classes=2, dropout=DROPOUT).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)
        except Exception as e:
            print(f"  HATA (model yüklenemedi): {e}")
            continue

        probs, labels = get_probs(model, test_loader)
        far, frr, aer, eer, auc = compute_biometrics(probs, labels)
        print(f"  FAR={far*100:.2f}%  FRR={frr*100:.2f}%  AER={aer*100:.2f}%  EER={eer*100:.2f}%  AUC={auc:.4f}")

        rows.append([ds, sc, backbone, finetune,
                     f"{far*100:.2f}", f"{frr*100:.2f}",
                     f"{aer*100:.2f}", f"{eer*100:.2f}", f"{auc:.4f}"])

    # CSV yaz
    out_csv = os.path.join(BIO_DIR, "biometric_all.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "scenario", "backbone", "finetune",
                    "FAR_%", "FRR_%", "AER_%", "EER_%", "AUC"])
        w.writerows(rows)

    print(f"\nTüm biyometrik metrikler: {out_csv}")
    print(f"Toplam {len(rows)} model işlendi.")


if __name__ == "__main__":
    main()
