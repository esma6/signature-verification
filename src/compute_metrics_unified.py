"""
ORTAK BİYOMETRİK METRİK SCRIPTİ - compute_metrics_unified.py
=============================================================
Altı eğitilmiş modeli (3 veri seti × 2 senaryo) kendi test setlerinde
değerlendirir, biyometrik metrikleri (FAR, FRR, AER, EER) hesaplar,
her veri seti için WD-WI karşılaştırma grafiği üretir ve hepsini
tek bir özet JSON + özet tabloda toplar.

ÖNEMLİ: train_unified.py ile AYNI bölme mantığını (signature_data) kullanır,
böylece her model tam da eğitildiği test seti üzerinde değerlendirilir.

Kullanım:
    python compute_metrics_unified.py
    (tüm mevcut outputs_unified/<ds>_<sc>/best_model.pth'leri tarar)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

import signature_data as sigdata

OUTPUT_DIR = "outputs_unified"
METRICS_DIR = os.path.join(OUTPUT_DIR, "_metrics")
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
TEST_RATIO = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = ["cedar", "bhsig", "utsig"]
SCENARIOS = ["wd", "wi"]
DS_LABEL = {"cedar": "CEDAR (Latin)", "bhsig": "BHSig (Devanagari)", "utsig": "UTSig (Perso-Arabic)"}


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


def val_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(num_classes=2, dropout=0.3):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )
    return model


def load_model(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    dropout = ckpt.get("dropout", 0.3)
    model = build_model(num_classes=len(ckpt["class_names"]), dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def collect_scores(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    scores, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE)
        probs = torch.softmax(model(imgs), dim=1)[:, 1]  # genuine olasılığı
        scores.extend(probs.cpu().numpy())
        labels.extend(lbls.numpy())
    return np.array(scores), np.array(labels)


def far_frr(scores, labels, thr):
    genuine = scores[labels == 1]
    forged = scores[labels == 0]
    far = float(np.mean(forged >= thr)) if len(forged) else 0.0
    frr = float(np.mean(genuine < thr)) if len(genuine) else 0.0
    return far, frr


def compute_eer(scores, labels):
    ths = np.linspace(0, 1, 1001)
    fars, frrs = [], []
    for t in ths:
        f, r = far_frr(scores, labels, t)
        fars.append(f); frrs.append(r)
    fars, frrs = np.array(fars), np.array(frrs)
    idx = int(np.argmin(np.abs(fars - frrs)))
    return (fars[idx] + frrs[idx]) / 2.0, fars[idx], frrs[idx], ths[idx]


def all_metrics(scores, labels):
    far05, frr05 = far_frr(scores, labels, 0.5)
    eer, fe, re_, th = compute_eer(scores, labels)
    return {
        "FAR@0.5": far05, "FRR@0.5": frr05, "AER@0.5": (far05 + frr05) / 2.0,
        "EER": eer, "FAR@EER": fe, "FRR@EER": re_, "EER_threshold": th,
    }


def plot_ds_comparison(ds, wd, wi, path):
    keys = ["FAR@0.5", "FRR@0.5", "AER@0.5", "EER", "FAR@EER", "FRR@EER"]
    wdv = [wd[k] * 100 for k in keys]
    wiv = [wi[k] * 100 for k in keys]
    x = np.arange(len(keys)); w = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - w/2, wdv, w, label="Writer-Dependent", color="#1f3a5f", edgecolor="black")
    b2 = ax.bar(x + w/2, wiv, w, label="Writer-Independent", color="#cc0000", edgecolor="black")
    ax.set_ylabel("Hata Oranı (%)"); ax.set_xlabel("Metrik")
    ax.set_title(f"{DS_LABEL[ds]} — Biyometrik Metrik Karşılaştırması", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(keys); ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def plot_overall_eer(results, path):
    """Üç veri seti × WD/WI EER'lerini tek grafikte göster."""
    x = np.arange(len(DATASETS)); w = 0.35
    wd = [results[d]["wd"]["EER"] * 100 for d in DATASETS]
    wi = [results[d]["wi"]["EER"] * 100 for d in DATASETS]
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, wd, w, label="Writer-Dependent", color="#1f3a5f", edgecolor="black")
    b2 = ax.bar(x + w/2, wi, w, label="Writer-Independent", color="#cc0000", edgecolor="black")
    ax.set_ylabel("EER (%)"); ax.set_title("Veri Setleri Arası EER Karşılaştırması", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"%{h:.2f}", (bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    print(f"Cihaz: {DEVICE}\n")
    results = {d: {} for d in DATASETS}
    vt = val_transform(IMG_SIZE)

    for ds in DATASETS:
        for sc in SCENARIOS:
            tag = f"{ds}_{sc}"
            mpath = os.path.join(OUTPUT_DIR, tag, "best_model.pth")
            if not os.path.exists(mpath):
                print(f"[ATLA] Model yok: {mpath}")
                continue
            print(f"[{tag}] değerlendiriliyor...")
            split = sigdata.make_split(ds, sc, test_ratio=TEST_RATIO, seed=SEED)
            test_ds = ListDataset(split["test_samples"], transform=vt)
            model = load_model(mpath)
            scores, labels = collect_scores(model, test_ds)
            results[ds][sc] = all_metrics(scores, labels)

    # Konsol özeti
    print("\n" + "=" * 78)
    print(f"{'Veri Seti / Senaryo':<26}{'FAR@0.5':>10}{'FRR@0.5':>10}{'AER@0.5':>10}{'EER':>10}")
    print("=" * 78)
    for ds in DATASETS:
        for sc in SCENARIOS:
            if sc not in results[ds]:
                continue
            m = results[ds][sc]
            print(f"{DS_LABEL[ds]+' / '+sc.upper():<26}"
                  f"{m['FAR@0.5']*100:>9.2f}%{m['FRR@0.5']*100:>9.2f}%"
                  f"{m['AER@0.5']*100:>9.2f}%{m['EER']*100:>9.2f}%")
    print("=" * 78)

    # Grafikler
    for ds in DATASETS:
        if "wd" in results[ds] and "wi" in results[ds]:
            plot_ds_comparison(ds, results[ds]["wd"], results[ds]["wi"],
                               os.path.join(METRICS_DIR, f"comparison_{ds}.png"))
    if all("wd" in results[d] and "wi" in results[d] for d in DATASETS):
        plot_overall_eer(results, os.path.join(METRICS_DIR, "overall_eer.png"))

    with open(os.path.join(METRICS_DIR, "all_biometric_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nÖzet JSON ve grafikler: {METRICS_DIR}/")


if __name__ == "__main__":
    main()
