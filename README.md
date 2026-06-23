# Offline Signature Verification Across Writing Systems

A fair, controlled comparison of offline handwritten signature verification across **three writing systems** — Latin (CEDAR), Devanagari (BHSig), and Perso-Arabic (UTSig) — using **transfer learning with four pre-trained backbones** under one identical protocol, plus a **frozen-vs-fine-tuned ablation**.

> 🇹🇷 Türkçe açıklama için [README.tr.md](README.tr.md) dosyasına bakın.

---

## Motivation

Most offline signature verification studies evaluate on a **single dataset** (usually Latin-script CEDAR), with a **single architecture**, and rarely test whether reported performance generalizes to other writing systems. Because methodological choices differ from paper to paper, it is hard to tell whether a performance gap reflects the **intrinsic difficulty of the data**, the **choice of architecture**, or just differences in method.

This project removes that ambiguity. **Every experiment shares one identical protocol** — same hyperparameters, same code, same random seed. The only variables are *which dataset*, *which scenario* (writer-dependent vs writer-independent), *which backbone*, and (for the ablation) *whether the backbone is frozen or partially fine-tuned*.

## What's new in this version

The study was extended from a single-backbone baseline into a full comparative study:

- **Four backbones** compared under the identical protocol: ResNet50, VGG19, DenseNet121, EfficientNet-B0.
- **Ablation study** on the best backbone (VGG19): frozen backbone vs last-block fine-tuning.
- **Biometric evaluation** (FAR/FRR/AER/EER) computed for **all 30 models**, not just accuracy.
- **30 models total**: 4 backbones × 3 datasets × 2 scenarios (24) + VGG19 ablation × 6.

![Architecture](docs/architecture_multibackbone.png)

## Key Idea

- **Frozen backbone** (ImageNet pre-trained): only the classifier head is trained — a fair, low-variance baseline for cross-dataset comparison.
- **Two scenarios per dataset:**
  - **Writer-Dependent (WD):** random split; the same writer may appear in both train and test.
  - **Writer-Independent (WI):** writer-disjoint split; test writers are never seen in training.
- **Ablation:** on VGG19, the last convolutional block is unfrozen (fine-tuned at a smaller lr=1e-5) to measure how much the frozen-backbone design decision costs or gains.
- **Biometric evaluation:** beyond accuracy/precision/recall/F1, we report FAR, FRR, AER, and EER for every model.

## Results — stage by stage

### Stage 1 — Baseline: single backbone (ResNet50)

The original study used a frozen ResNet50. It established the central finding — a clear **difficulty ordering by writing system, CEDAR < BHSig < UTSig** — but left open whether the ordering was an artifact of the chosen architecture.

![Stage 1](results/figures/stage1_resnet_eer.png)

### Stage 2 — Multi-backbone comparison

Adding three more backbones under the identical protocol reveals that **architecture matters enormously on the hard, non-Latin datasets**. VGG19 wins 4 of 6 scenarios and is dramatically better where it counts: on BHSig-WD it cuts EER from 24.5% (ResNet50) to **4.45%**, and on UTSig-WD from 26.5% to **10.27%**.

![Stage 2 — accuracy](results/figures/backbone_accuracy.png)

![Stage 2 — EER](results/figures/stage2_backbone_eer.png)

**Classification accuracy (frozen backbone):**

| Dataset / Scenario | ResNet50 | VGG19 | DenseNet121 | EfficientNet-B0 |
|--------------------|---------:|------:|------------:|----------------:|
| CEDAR-WD | 0.945 | **0.977** | 0.962 | 0.970 |
| CEDAR-WI | 0.866 | 0.915 | 0.879 | **0.917** |
| BHSig-WD | 0.772 | **0.958** | 0.799 | 0.754 |
| BHSig-WI | 0.808 | **0.823** | 0.818 | 0.790 |
| UTSig-WD | 0.756 | **0.906** | 0.734 | 0.741 |
| UTSig-WI | 0.728 | 0.663 | **0.735** | 0.701 |

VGG19 dominates except on the hardest writer-independent cases — on UTSig-WI it actually **overfits** (recall drops to 0.33), and DenseNet121 takes the lead. High capacity is not universally best.

### Stage 3 — Ablation: frozen vs fine-tuned (VGG19)

Unfreezing the last block helps mainly in **writer-dependent** scenarios and is unreliable for writer-independent ones — confirming that the frozen-backbone choice is a sound, robust baseline.

![Stage 3 — EER](results/figures/stage3_ablation_eer.png)

| Dataset / Scenario | Frozen EER | Fine-tuned EER | Δ EER |
|--------------------|-----------:|---------------:|------:|
| CEDAR-WD | 2.46 | **0.76** | −1.70 |
| CEDAR-WI | 8.90 | 8.90 | 0.00 |
| BHSig-WD | 4.45 | **2.28** | −2.17 |
| BHSig-WI | 17.80 | **15.95** | −1.85 |
| UTSig-WD | 10.27 | **7.43** | −2.84 |
| UTSig-WI | 37.73 | 35.10 | −2.63 |

Fine-tuning lowers EER in writer-dependent settings (e.g. UTSig-WD 10.27→7.43), but in writer-independent UTSig-WI the model stays in a severe overfitting regime (FRR ~60%, rejecting most genuine signatures).

### Main finding

A clear **difficulty ordering by writing system** holds across all four backbones: **CEDAR < BHSig < UTSig**. The dataset effect dominates the WD–WI scenario effect, and high accuracy on a single dataset (or with a single architecture) is **not** by itself evidence of generalizability — the choice of backbone changes results by up to ~20 EER points on the hardest datasets.

## Datasets

| Property | CEDAR | BHSig | UTSig |
|----------|------:|------:|------:|
| Writing system | Latin | Devanagari | Perso-Arabic |
| Writers | 55 | 260 | 115 |
| Genuine | 1,320 | 6,240 | 3,105 |
| Forged | 1,320 | 7,800 | 4,830 |
| Total | 2,640 | 14,040 | 7,935 |
| Forgery type | Amateur | Skilled | Skilled |

> The datasets themselves are **not** redistributed here. Obtain them from their original sources and arrange each as `train/{genuine,forged}` and `test/{genuine,forged}`. Update the paths at the top of `src/signature_data.py`.

## Repository structure

```
.
├── src/
│   ├── signature_data.py            # dataset loading + WD/WI splitting (all 3 datasets)
│   ├── train_unified.py             # training: --dataset --scenario --backbone --finetune
│   ├── compute_metrics_unified.py   # biometric metrics for the original ResNet50 run
│   ├── compute_biometrics_all.py    # FAR/FRR/EER for ALL saved models
│   └── aggregate_results.py         # backbone + ablation comparison tables/figures
├── results/
│   ├── figures/                     # staged comparison charts (accuracy + EER)
│   └── metrics/                     # all_results_classification.csv, all_results_biometric.csv, ablation_vgg19.csv
├── docs/
│   └── architecture_multibackbone.png   # architecture + ablation diagram
├── paper/                           # manuscript (docx + pdf)
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch. A GPU is recommended but not required (these experiments were run on CPU).

## Usage

Edit dataset paths at the top of `src/signature_data.py`, then run experiments. Each is parameterized by dataset, scenario, backbone, and fine-tune mode:

```bash
cd src
# Multi-backbone comparison (frozen backbone)
python train_unified.py --dataset cedar --scenario wd --backbone vgg19
python train_unified.py --dataset bhsig --scenario wd --backbone densenet121
python train_unified.py --dataset utsig --scenario wi --backbone efficientnet_b0
# ... (4 backbones × 3 datasets × 2 scenarios)

# Ablation (frozen vs fine-tuned, on VGG19)
python train_unified.py --dataset utsig --scenario wd --backbone vgg19 --finetune last_block
```

Then aggregate results and compute biometric metrics for every saved model:

```bash
python aggregate_results.py          # comparison tables + figures
python compute_biometrics_all.py     # FAR/FRR/AER/EER for all 30 models
```

## Training protocol (identical for all experiments)

| Hyperparameter | Value |
|----------------|-------|
| Backbones | ResNet50 · VGG19 · DenseNet121 · EfficientNet-B0 (ImageNet, frozen) |
| Fine-tune (ablation) | last conv block unfrozen, lr=1e-5 |
| Optimizer | Adam, lr = 1e-4 (head) |
| Weight decay | 5e-4 |
| Batch size | 32 |
| Epochs | 30 (early stopping, patience = 5) |
| Dropout | 0.3 |
| Image size | 224 × 224 |
| Seed | 42 |

## Reproducibility

Splits are performed in memory with a fixed seed (42) and are fully deterministic — all scripts use the same `signature_data` module, so each model is always evaluated on exactly the test set it was trained against. `compute_biometrics_all.py` reloads each saved model and re-scores it on that same deterministic test set.

## Citation

If you use this code, please cite the accompanying manuscript (see `paper/`). A BibTeX entry will be added upon publication.

## License

Released under the MIT License — see [LICENSE](LICENSE).
