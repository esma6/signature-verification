"""
ORTAK VERİ MODÜLÜ - signature_data.py
======================================
Üç veri seti (CEDAR, BHSig, HYBRID) ve iki senaryo (WD, WI) için
TEK noktadan, bellekte (kopyalamasız), deterministik veri bölme.

Tasarım ilkesi: TÜM deneyler aynı kod ve aynı protokolle çalışır.
Tek değişkenler: (1) hangi veri seti, (2) WD mi WI mi.

- WD (writer-dependent): tüm imzalar tek havuz, kişiye bakmadan RASTGELE bölünür.
- WI (writer-independent): kişi bazında bölünür; test kişileri eğitimde hiç görülmez.

Writer ID çıkarımı dosya adından yapılır:
  CEDAR:  original_5_12.png / forgeries_5_12.png            -> 5
  BHSig:  BHSig260-Bengali_14_B-S-14-G-01.tif               -> 14 (+ dil öneki ile global)
  HYBRID: yukarıdakilerin birleşimi; her kişiye veri-seti-önekli GLOBAL id verilir
          ("CEDAR:5", "BHSIG-Bengali:14") -> çakışma olmaz.
"""

import re
import random
from pathlib import Path
from collections import defaultdict

# ---- Kaynak yolları (üçü de Downloads altında, hepsi dengeli/sağlam) ----
CEDAR_DIR = r"C:\Users\ETU\Downloads\veri\CEDAR_split"
BHSIG_DIR = r"C:\Users\ETU\Downloads\veri\BHSig_split"
UTSIG_DIR = r"C:\Users\ETU\Downloads\veri\UTSig_split"

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_TO_IDX = {"forged": 0, "genuine": 1}


# =========================================================
# Writer ID çıkarımı (veri setine göre)
# =========================================================
def cedar_writer_id(filename):
    m = re.match(r"(?:original|forgeries)_(\d+)_\d+\.\w+", filename, re.IGNORECASE)
    return f"CEDAR:{int(m.group(1))}" if m else None


def bhsig_writer_id(filename):
    """
    'bhsig_test_BHSig260-Bengali_14_B-S-14-G-01.tif' veya
    'BHSig260-Hindi_12_H-S-12-G-03.tif' -> dil + id ile global.
    """
    m = re.search(r"BHSig\d*-(\w+?)_(\d+)_", filename, re.IGNORECASE)
    if m:
        lang, wid = m.group(1), int(m.group(2))
        return f"BHSIG-{lang}:{wid}"
    m = re.search(r"-S-(\d+)-", filename, re.IGNORECASE)
    if m:
        return f"BHSIG:{int(m.group(1))}"
    return None


def utsig_writer_id(filename):
    """
    UTSig dosya adından kişi ID'si. Writer ID daima 'UTS_' den hemen
    sonraki sayıdır; ortadaki etiket (genuine'de yok, forged'da 'Simple'
    / 'Skilled' vb.) değişebilir:
      genuine: 'UTS_102_1.tif'         -> 102
      forged:  'UTS_92_Simple_14.tif'  -> 92
      forged:  'UTS_5_Skilled_3.tif'   -> 5
    Sadece UTS_ ile başlayanları kabul eder; yabancı dosyaları atlar.
    """
    m = re.match(r"UTS_(\d+)_", filename, re.IGNORECASE)
    if m:
        return f"UTSIG:{int(m.group(1))}"
    return None  # UTS_ değilse atla


# =========================================================
# Dosya toplama
# =========================================================
def _scan_dataset(source_dir, id_func):
    """
    Bir veri setinin train+test klasörlerini tarar.
    Döndürür: list of (path, label, writer_id)
    Deterministik: split/sınıf/dosya-adı sıralı.
    """
    source_dir = Path(source_dir)
    items = []
    for split in ["train", "test"]:
        for cls in ["genuine", "forged"]:
            folder = source_dir / split / cls
            if not folder.exists():
                continue
            for img_path in sorted(folder.glob("*"), key=lambda p: p.name):
                if img_path.suffix.lower() not in IMG_EXT:
                    continue
                wid = id_func(img_path.name)
                if wid is None:
                    continue
                items.append((str(img_path), CLASS_TO_IDX[cls], wid))
    return items


def collect_items(dataset):
    """dataset: 'cedar' | 'bhsig' | 'utsig' -> list of (path, label, writer_id)"""
    ds = dataset.lower()
    if ds == "cedar":
        return _scan_dataset(CEDAR_DIR, cedar_writer_id)
    if ds == "bhsig":
        return _scan_dataset(BHSIG_DIR, bhsig_writer_id)
    if ds == "utsig":
        return _scan_dataset(UTSIG_DIR, utsig_writer_id)
    raise ValueError(f"Bilinmeyen veri seti: {dataset}")


# =========================================================
# Bölme
# =========================================================
def split_writer_dependent(items, test_ratio, seed):
    """RASTGELE bölme (kişiye bakmadan). Aynı kişi her iki tarafta olabilir."""
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_ratio)))
    test = [(p, l) for (p, l, w) in shuffled[:n_test]]
    train = [(p, l) for (p, l, w) in shuffled[n_test:]]
    return train, test


def split_writer_independent(items, test_ratio, seed):
    """KİŞİ bazında bölme. Test kişileri eğitimde hiç görülmez."""
    writers = sorted(set(w for (_, _, w) in items))
    rng = random.Random(seed)
    shuffled = writers.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_ratio)))
    test_writers = set(shuffled[:n_test])
    train = [(p, l) for (p, l, w) in items if w not in test_writers]
    test = [(p, l) for (p, l, w) in items if w in test_writers]
    return train, test, sorted(test_writers), len(writers)


def make_split(dataset, scenario, test_ratio=0.2, seed=42):
    """
    Ana giriş noktası.
    dataset:  'cedar' | 'bhsig' | 'utsig'
    scenario: 'wd' | 'wi'
    Döndürür: dict {train_samples, test_samples, info(str)}
    """
    items = collect_items(dataset)
    if not items:
        raise RuntimeError(f"Hiç dosya bulunamadı: {dataset}")

    n_g = sum(1 for _, l, _ in items if l == 1)
    n_f = sum(1 for _, l, _ in items if l == 0)
    n_writers = len(set(w for _, _, w in items))

    if scenario.lower() == "wd":
        train, test = split_writer_dependent(items, test_ratio, seed)
        info = (f"[{dataset.upper()} / WD] havuz={len(items)} "
                f"({n_g} genuine + {n_f} forged), kişi={n_writers}\n"
                f"  rastgele bölme -> train={len(train)}, test={len(test)}")
    elif scenario.lower() == "wi":
        train, test, test_writers, total_w = split_writer_independent(items, test_ratio, seed)
        info = (f"[{dataset.upper()} / WI] havuz={len(items)} "
                f"({n_g} genuine + {n_f} forged), kişi={total_w}\n"
                f"  kişi bazında bölme -> train={len(train)}, test={len(test)}, "
                f"test kişi sayısı={len(test_writers)}")
    else:
        raise ValueError(f"Bilinmeyen senaryo: {scenario}")

    return {"train_samples": train, "test_samples": test, "info": info}


if __name__ == "__main__":
    # Hızlı doğrulama: tüm kombinasyonların özetini yazdır (sadece kişisel makinede çalışır)
    for ds in ["cedar", "bhsig", "utsig"]:
        for sc in ["wd", "wi"]:
            try:
                r = make_split(ds, sc)
                print(r["info"]); print()
            except Exception as e:
                print(f"[{ds}/{sc}] HATA: {e}\n")
