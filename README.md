# Offline Signature Verification Across Writing Systems

A controlled comparison of offline handwritten signature verification across **three writing systems** — Latin (CEDAR), Bengali + Devanagari (BHSig260), and Perso-Arabic (UTSig) — using **transfer learning with four ImageNet-pretrained CNN backbones** under one identical protocol, plus a **frozen-vs-last-block fine-tuning ablation** on VGG19.

> 🇹🇷 Türkçe açıklama için [README.tr.md](README.tr.md) dosyasına bakın.

---

## Motivation

Most offline signature verification studies evaluate a method on a **single dataset**, often with a **single architecture** and a single evaluation protocol. This makes it difficult to determine whether a reported performance gap is caused by the intrinsic difficulty of the dataset, the writing system, the model architecture, or the evaluation scenario.

This project addresses that problem through a unified experimental design. Every experiment follows the same pipeline, uses the same codebase, the same preprocessing procedure, the same random seed, and the same training protocol. The controlled variables are:

* the dataset / writing system,
* the evaluation scenario: writer-dependent (WD) or writer-independent (WI),
* the CNN backbone,
* and, for the ablation study, whether VGG19 is kept frozen or partially fine-tuned.

The aim is not only to report high accuracy, but to examine how robust transfer learning is when the writing system, writer split protocol, and backbone architecture change.

---

## What is new in this version?

The study was extended from a single-backbone baseline into a full comparative transfer learning study:

* **Three datasets / writing systems:** CEDAR, BHSig260, and UTSig.
* **Two evaluation scenarios per dataset:** writer-dependent (WD) and writer-independent (WI).
* **Four frozen ImageNet-pretrained CNN backbones:** ResNet50, VGG19, DenseNet121, and EfficientNet-B0.
* **VGG19 ablation:** frozen backbone vs last-block fine-tuning.
* **Biometric evaluation for all models:** FAR, FRR, AER, EER, and ROC-AUC.
* **30 model runs in total:**
  4 backbones × 3 datasets × 2 scenarios = 24 frozen-backbone runs
  plus 6 VGG19 fine-tuning ablation runs.

![Architecture](docs/architecture_multibackbone.png)

---

## Key idea

This repository evaluates transfer learning for offline signature verification under a unified and reproducible protocol.

### Frozen-backbone transfer learning

Each CNN backbone is initialized with ImageNet-pretrained weights. In the main comparison, the convolutional backbone is frozen and only the classifier head is trained. This design provides a fair and low-variance baseline for comparing datasets, writing systems, scenarios, and architectures.

### Writer-dependent and writer-independent evaluation

Each dataset is evaluated under two scenarios:

* **Writer-Dependent (WD):**
  A random image-level split is used. The same writer may appear in both training and test sets. This setting is easier and may produce optimistic performance estimates.

* **Writer-Independent (WI):**
  A writer-disjoint split is used. Test writers are never seen during training. This setting is more realistic and tests the model’s ability to generalize to unseen writers.

### VGG19 ablation

After the frozen-backbone comparison, VGG19 is selected for an ablation study because it shows the strongest overall performance across most scenarios. In the ablation, the last convolutional block is unfrozen and fine-tuned with a smaller learning rate (`lr=1e-5`) while keeping the rest of the protocol unchanged.

### Biometric metrics

In addition to classification metrics such as accuracy, precision, recall, F1, and ROC-AUC, the project reports biometric metrics that are more informative for signature verification:

* **FAR:** False Acceptance Rate
* **FRR:** False Rejection Rate
* **AER:** Average Error Rate
* **EER:** Equal Error Rate

---

## Results

### Stage 1 — Baseline: frozen ResNet50

The initial baseline used a frozen ResNet50 backbone. This baseline showed that performance varies considerably across writing systems and evaluation scenarios, but it left open an important question: are these differences caused by the dataset itself, or by the chosen backbone?

![Stage 1](results/figures/stage1_resnet_eer.png)

---

### Stage 2 — Multi-backbone comparison

Adding VGG19, DenseNet121, and EfficientNet-B0 under the same protocol shows that **backbone choice strongly affects performance**, especially on more challenging non-Latin datasets.

![Stage 2 — accuracy](results/figures/backbone_accuracy.png)

![Stage 2 — EER](results/figures/stage2_backbone_eer.png)

#### Classification metrics: frozen backbone

| Dataset / Scenario | Backbone        |       Acc |      Prec |       Rec |        F1 |       AUC |
| ------------------ | --------------- | --------: | --------: | --------: | --------: | --------: |
| CEDAR-WD           | ResNet50        |     0.945 |     0.914 |     0.981 |     0.946 |     0.991 |
| CEDAR-WD           | VGG19           | **0.977** | **0.963** | **0.992** | **0.977** | **0.996** |
| CEDAR-WD           | DenseNet121     |     0.962 |     0.958 |     0.966 |     0.962 |     0.996 |
| CEDAR-WD           | EfficientNet-B0 |     0.970 |     0.966 |     0.973 |     0.970 |     0.994 |
| CEDAR-WI           | ResNet50        |     0.866 |     0.832 |     0.917 |     0.872 |     0.938 |
| CEDAR-WI           | VGG19           |     0.915 |     0.954 |     0.871 |     0.911 |     0.976 |
| CEDAR-WI           | DenseNet121     |     0.879 |     0.817 |     0.977 |     0.890 |     0.985 |
| CEDAR-WI           | EfficientNet-B0 | **0.917** |     0.882 |     0.962 | **0.920** |     0.981 |
| BHSig-WD           | ResNet50        |     0.772 |     0.770 |     0.675 |     0.719 |     0.831 |
| BHSig-WD           | VGG19           | **0.958** | **0.961** | **0.943** | **0.952** | **0.992** |
| BHSig-WD           | DenseNet121     |     0.799 |     0.785 |     0.739 |     0.761 |     0.872 |
| BHSig-WD           | EfficientNet-B0 |     0.754 |     0.753 |     0.642 |     0.693 |     0.820 |
| BHSig-WI           | ResNet50        |     0.808 |     0.812 |     0.740 |     0.775 |     0.880 |
| BHSig-WI           | VGG19           | **0.823** |     0.804 | **0.794** | **0.799** |     0.885 |
| BHSig-WI           | DenseNet121     |     0.818 |     0.799 |     0.791 |     0.795 | **0.892** |
| BHSig-WI           | EfficientNet-B0 |     0.790 | **0.842** |     0.650 |     0.734 |     0.868 |
| UTSig-WD           | ResNet50        |     0.756 |     0.724 |     0.673 |     0.697 |     0.827 |
| UTSig-WD           | VGG19           | **0.906** | **0.927** | **0.842** | **0.882** | **0.961** |
| UTSig-WD           | DenseNet121     |     0.734 |     0.694 |     0.647 |     0.670 |     0.797 |
| UTSig-WD           | EfficientNet-B0 |     0.741 |     0.695 |     0.677 |     0.686 |     0.812 |
| UTSig-WI           | ResNet50        |     0.728 |     0.665 |     0.614 | **0.638** | **0.794** |
| UTSig-WI           | VGG19           |     0.663 |     0.633 |     0.330 |     0.434 |     0.690 |
| UTSig-WI           | DenseNet121     | **0.735** | **0.698** |     0.568 |     0.626 |     0.788 |
| UTSig-WI           | EfficientNet-B0 |     0.701 |     0.617 | **0.620** |     0.619 |     0.749 |

VGG19 gives the strongest or near-strongest performance in most scenarios, especially in writer-dependent settings. However, this advantage is not universal. In the UTSig-WI scenario, VGG19 performs poorly, with recall dropping to 0.330. This suggests a serious generalization loss under the most challenging writer-independent Perso-Arabic setting. The result may be related to overfitting, representation mismatch, or both.

---

### Stage 3 — Ablation: frozen VGG19 vs last-block fine-tuning

The ablation study compares frozen VGG19 against VGG19 with the last convolutional block unfrozen and fine-tuned.

![Stage 3 — EER](results/figures/stage3_ablation_eer.png)

#### VGG19 ablation results

| Dataset / Scenario | Acc Frozen | Acc Fine-tuned |  Δ Acc | AUC Frozen | AUC Fine-tuned |
| ------------------ | ---------: | -------------: | -----: | ---------: | -------------: |
| CEDAR-WD           |      0.977 |      **0.991** | +0.014 |      0.996 |      **1.000** |
| CEDAR-WI           |  **0.915** |          0.901 | −0.014 |  **0.976** |          0.967 |
| BHSig-WD           |      0.958 |      **0.978** | +0.020 |      0.992 |      **0.998** |
| BHSig-WI           |      0.823 |      **0.842** | +0.019 |      0.885 |      **0.901** |
| UTSig-WD           |      0.906 |      **0.927** | +0.021 |      0.961 |      **0.973** |
| UTSig-WI           |      0.663 |      **0.687** | +0.024 |      0.690 |      **0.718** |

#### Biometric ablation results

| Dataset / Scenario | Setup      |        FAR |        FRR |        AER |        EER |       AUC |
| ------------------ | ---------- | ---------: | ---------: | ---------: | ---------: | --------: |
| CEDAR-WD           | Frozen     |      3.75% |      0.77% |      2.26% |      2.46% |     0.996 |
| CEDAR-WD           | Fine-tuned |  **1.87%** |  **0.00%** |  **0.94%** |  **0.76%** | **1.000** |
| CEDAR-WI           | Frozen     |  **4.17%** |     12.88% |  **8.52%** |      8.90% | **0.976** |
| CEDAR-WI           | Fine-tuned |      8.33% | **11.36%** |      9.85% |      8.90% |     0.967 |
| BHSig-WD           | Frozen     |      2.95% |      5.75% |      4.35% |      4.45% |     0.992 |
| BHSig-WD           | Fine-tuned |  **1.19%** |  **3.45%** |  **2.32%** |  **2.28%** | **0.998** |
| BHSig-WI           | Frozen     | **15.45%** |     20.59% |     18.02% |     17.80% |     0.885 |
| BHSig-WI           | Fine-tuned |     15.58% | **16.19%** | **15.88%** | **15.95%** | **0.901** |
| UTSig-WD           | Frozen     |  **4.76%** |     15.84% |     10.30% |     10.27% |     0.961 |
| UTSig-WD           | Fine-tuned |      7.25% |  **7.39%** |  **7.32%** |  **7.43%** | **0.973** |
| UTSig-WI           | Frozen     | **12.32%** |     66.99% |     39.65% |     37.73% |     0.690 |
| UTSig-WI           | Fine-tuned |     13.25% | **59.42%** | **36.34%** | **35.10%** | **0.718** |

Last-block fine-tuning generally improves EER and AUC, especially in writer-dependent settings. However, it does not guarantee better generalization. In CEDAR-WI, accuracy and AUC decrease while EER remains unchanged. In UTSig-WI, fine-tuning improves EER slightly, but the model still rejects a large proportion of genuine signatures.

---

## Main findings

The main conclusion is that offline signature verification performance is highly sensitive to the interaction between **writing system**, **writer split protocol**, **CNN backbone**, and **fine-tuning strategy**.

Key observations:

* VGG19 is the strongest overall backbone in most scenarios, but not in all.
* Writer-independent evaluation is substantially more challenging because the model must generalize to unseen writers.
* UTSig-WI is the most difficult setting in this study, especially for VGG19.
* Fine-tuning the last block helps in several scenarios, but its benefit is conditional and can be unreliable under writer-independent evaluation.
* High performance on a single dataset or a single split protocol is not sufficient evidence of generalizability.

---

## Datasets

| Property           |   CEDAR |             BHSig260 |        UTSig |
| ------------------ | ------: | -------------------: | -----------: |
| Writing system     |   Latin | Bengali + Devanagari | Perso-Arabic |
| Writers            |      55 |                  260 |          115 |
| Genuine signatures |   1,320 |                6,240 |        3,105 |
| Forged signatures  |   1,320 |                7,800 |        4,830 |
| Total used         |   2,640 |               14,040 |        7,935 |
| Forgery type       | Skilled |              Skilled |      Skilled |

For UTSig, opposite-hand samples were excluded. Only genuine and skilled-forgery signatures were used. Therefore, the number of used samples is:

```text
115 × (27 genuine + 42 skilled forgeries) = 7,935
```

The datasets themselves are **not redistributed** in this repository. Please obtain them from their original sources and configure the dataset paths in `src/signature_data.py`.

---

## Repository structure

```text
.
├── src/
│   ├── signature_data.py            # dataset loading and WD/WI splitting
│   ├── train_unified.py             # training: --dataset --scenario --backbone --finetune
│   ├── compute_biometrics_all.py    # FAR, FRR, AER, EER for all saved models
│   └── aggregate_results.py         # result aggregation and figure generation
├── results/
│   ├── figures/                     # accuracy, EER, and ablation figures
│   └── metrics/                     # classification and biometric result tables
├── docs/
│   └── architecture_multibackbone.png
├── paper/                           # manuscript files
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

The code requires Python 3.9+ and PyTorch. A GPU is recommended, but the experiments can also be run on CPU.

---

## Usage

Edit dataset paths at the top of `src/signature_data.py`, then run the experiments.

### Frozen-backbone comparison

```bash
cd src

python train_unified.py --dataset cedar --scenario wd --backbone resnet50
python train_unified.py --dataset cedar --scenario wd --backbone vgg19
python train_unified.py --dataset cedar --scenario wd --backbone densenet121
python train_unified.py --dataset cedar --scenario wd --backbone efficientnet_b0

python train_unified.py --dataset bhsig260 --scenario wi --backbone vgg19
python train_unified.py --dataset utsig --scenario wi --backbone densenet121
```

Run all combinations for:

```text
4 backbones × 3 datasets × 2 scenarios = 24 frozen-backbone experiments
```

### VGG19 last-block fine-tuning ablation

```bash
python train_unified.py --dataset cedar --scenario wd --backbone vgg19 --finetune last_block
python train_unified.py --dataset bhsig260 --scenario wi --backbone vgg19 --finetune last_block
python train_unified.py --dataset utsig --scenario wi --backbone vgg19 --finetune last_block
```

### Aggregate results

```bash
python aggregate_results.py
python compute_biometrics_all.py
```

---

## Training protocol

| Hyperparameter            | Value                                            |
| ------------------------- | ------------------------------------------------ |
| Backbones                 | ResNet50, VGG19, DenseNet121, EfficientNet-B0    |
| Pretraining               | ImageNet                                         |
| Main setting              | Frozen convolutional backbone                    |
| Classifier head           | Dropout 0.3 → FC 256 → ReLU → Dropout 0.3 → FC 2 |
| Fine-tuning ablation      | VGG19 last convolutional block unfrozen          |
| Fine-tuning learning rate | 1e-5                                             |
| Optimizer                 | Adam                                             |
| Head learning rate        | 1e-4                                             |
| Weight decay              | 5e-4                                             |
| Batch size                | 32                                               |
| Epochs                    | 30                                               |
| Early stopping            | patience = 5                                     |
| Image size                | 224 × 224                                        |
| Normalization             | ImageNet statistics                              |
| Random seed               | 42                                               |

---

## Reproducibility

All splits are deterministic and use the same random seed (`42`). The same dataset loading and splitting module is used for all backbones and scenarios, ensuring that models are evaluated on the corresponding deterministic hold-out split.

In the writer-independent setting, the split is writer-disjoint: test writers are never included in the training writers. This prevents identity leakage across training and test sets.

---

## Limitations

This repository reports a controlled single-seed experimental setup. Although this improves reproducibility, it does not fully characterize variance across random splits. Future work should include repeated runs with different seeds and k-fold or multi-split evaluation.

Another limitation is that early stopping uses the hold-out evaluation split rather than a fully separate validation set. A separate validation set would provide a cleaner separation between model selection and final testing.

Finally, the UTSig-WI failure mode requires deeper analysis. The observed performance drop may be caused by overfitting, representation mismatch between ImageNet features and Perso-Arabic signatures, limited writer diversity, or a combination of these factors.

---

## Citation

If you use this code or results, please cite the accompanying manuscript in `paper/`.

A BibTeX entry will be added after publication.

---

## License

Released under the MIT License. See [LICENSE](LICENSE).
