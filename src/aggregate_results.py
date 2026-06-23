"""
ÇOK-MODEL & ABLASYON ÖZET SCRIPTİ - aggregate_results.py
=========================================================
train_unified.py'nin ürettiği tüm outputs_unified/<...>/test_metrics.json
dosyalarını tarar ve iki analizi tablolar + grafikler halinde toplar:

  (A) ÇOKLU OMURGA KARŞILAŞTIRMASI  (finetune=none olan tüm koşular)
      -> her (veri seti, senaryo) için omurgaların yan yana karşılaştırması
      -> backbone_comparison.csv  + her veri seti için bar grafik

  (B) ABLASYON: donuk vs ince-ayar  (aynı omurga, finetune none vs last_block)
      -> ablation.csv  + bar grafik (Δ doğruluk)

Kullanım:
    python aggregate_results.py
    (önce train_unified.py ile istediğiniz koşuları çalıştırın)

Not: Bu script torch GEREKTİRMEZ; sadece JSON okur. Hızlı ve bağımsızdır.
"""

import os
import json
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs_unified"
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "_summary")
os.makedirs(SUMMARY_DIR, exist_ok=True)

DS_LABEL = {"cedar": "CEDAR (Latin)", "bhsig": "BHSig (Devanagari)",
            "utsig": "UTSig (Perso-Arabic)"}
DS_ORDER = ["cedar", "bhsig", "utsig"]
SC_ORDER = ["wd", "wi"]
BB_ORDER = ["resnet50", "vgg19", "densenet121", "efficientnet_b0"]


def load_all():
    """Tüm test_metrics.json kayıtlarını liste olarak döndürür."""
    records = []
    if not os.path.isdir(OUTPUT_DIR):
        print(f"UYARI: '{OUTPUT_DIR}' klasörü yok. Önce train_unified.py çalıştırın.")
        return records
    for name in sorted(os.listdir(OUTPUT_DIR)):
        mpath = os.path.join(OUTPUT_DIR, name, "test_metrics.json")
        if os.path.isfile(mpath):
            with open(mpath, encoding="utf-8") as f:
                rec = json.load(f)
            # Eski (backbone'suz) kayıtlar için varsayılan değerler
            rec.setdefault("backbone", "resnet50")
            rec.setdefault("finetune", "none")
            records.append(rec)
    return records


def write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ============================================================
# (A) ÇOKLU OMURGA KARŞILAŞTIRMASI
# ============================================================
def backbone_comparison(records):
    # sadece donuk omurga (adil protokol) koşuları
    frozen = [r for r in records if r.get("finetune", "none") == "none"]
    if not frozen:
        print("Çoklu omurga karşılaştırması için kayıt yok.")
        return

    # index: (dataset, scenario, backbone) -> rec
    idx = {(r["dataset"], r["scenario"], r["backbone"]): r for r in frozen}
    backbones = [b for b in BB_ORDER if any(r["backbone"] == b for r in frozen)]

    rows = []
    for ds in DS_ORDER:
        for sc in SC_ORDER:
            for bb in backbones:
                r = idx.get((ds, sc, bb))
                if r:
                    rows.append([ds, sc, bb,
                                 f"{r['test_accuracy']:.4f}",
                                 f"{r['precision']:.4f}",
                                 f"{r['recall']:.4f}",
                                 f"{r['f1_score']:.4f}",
                                 f"{r['roc_auc']:.4f}"])
    write_csv(os.path.join(SUMMARY_DIR, "backbone_comparison.csv"), rows,
              ["dataset", "scenario", "backbone", "accuracy",
               "precision", "recall", "f1", "roc_auc"])
    print(f"[A] backbone_comparison.csv yazıldı ({len(rows)} satır).")

    # Grafik: her veri seti için, WD/WI birlikte, omurgalar yan yana (doğruluk)
    for ds in DS_ORDER:
        present = [(sc, bb) for sc in SC_ORDER for bb in backbones
                   if (ds, sc, bb) in idx]
        if not present:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.8 / max(1, len(backbones))
        x_base = range(len(SC_ORDER))
        for j, bb in enumerate(backbones):
            ys = []
            for sc in SC_ORDER:
                r = idx.get((ds, sc, bb))
                ys.append(r["test_accuracy"] if r else 0)
            xs = [x + j * width for x in x_base]
            bars = ax.bar(xs, ys, width=width, label=bb)
            for x, y in zip(xs, ys):
                if y > 0:
                    ax.text(x, y + 0.005, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks([x + (len(backbones) - 1) * width / 2 for x in x_base])
        ax.set_xticklabels([s.upper() for s in SC_ORDER])
        ax.set_ylabel("Test Doğruluğu")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Omurga Karşılaştırması — {DS_LABEL[ds]}")
        ax.legend(title="Omurga", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = os.path.join(SUMMARY_DIR, f"backbone_{ds}.png")
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"    grafik: {out}")


# ============================================================
# (B) ABLASYON: DONUK vs İNCE-AYAR
# ============================================================
def ablation(records):
    # aynı (ds, sc, backbone) için hem none hem last_block olanları eşle
    idx = {(r["dataset"], r["scenario"], r["backbone"], r["finetune"]): r
           for r in records}
    keys = set((r["dataset"], r["scenario"], r["backbone"]) for r in records)

    rows = []
    pairs = []  # grafik için
    for (ds, sc, bb) in sorted(keys):
        r_none = idx.get((ds, sc, bb, "none"))
        r_ft = idx.get((ds, sc, bb, "last_block"))
        if r_none and r_ft:
            d = r_ft["test_accuracy"] - r_none["test_accuracy"]
            rows.append([ds, sc, bb,
                         f"{r_none['test_accuracy']:.4f}",
                         f"{r_ft['test_accuracy']:.4f}",
                         f"{d:+.4f}"])
            pairs.append((f"{ds}-{sc}-{bb}",
                          r_none["test_accuracy"], r_ft["test_accuracy"]))

    if not rows:
        print("[B] Ablasyon için eşleşen (none + last_block) çift bulunamadı.")
        print("    İpucu: aynı omurgayı hem --finetune none hem --finetune last_block ile çalıştırın.")
        return

    write_csv(os.path.join(SUMMARY_DIR, "ablation.csv"), rows,
              ["dataset", "scenario", "backbone",
               "acc_frozen", "acc_finetuned", "delta"])
    print(f"[B] ablation.csv yazıldı ({len(rows)} satır).")

    # Grafik: donuk vs ince-ayar yan yana
    labels = [p[0] for p in pairs]
    frozen_acc = [p[1] for p in pairs]
    ft_acc = [p[2] for p in pairs]
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(labels)), 5))
    ax.bar([i - w / 2 for i in x], frozen_acc, width=w, label="Donuk omurga")
    ax.bar([i + w / 2 for i in x], ft_acc, width=w, label="İnce-ayar (son blok)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test Doğruluğu")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablasyon: Donuk Omurga vs Son-Blok İnce-Ayarı")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(SUMMARY_DIR, "ablation_frozen_vs_finetune.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"    grafik: {out}")


def main():
    records = load_all()
    if not records:
        return
    print(f"Toplam {len(records)} model kaydı okundu.\n")
    backbone_comparison(records)
    print()
    ablation(records)
    print(f"\nTüm özetler: {SUMMARY_DIR}/")


if __name__ == "__main__":
    main()
