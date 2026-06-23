# Yazı Sistemleri Arası Çevrimdışı İmza Doğrulama

Üç farklı **yazı sistemi** — Latin (CEDAR), Devanagari (BHSig) ve Perso-Arabik (UTSig) — üzerinde, **dört önceden eğitilmiş omurga ile transfer öğrenme** kullanılarak tek ve özdeş bir protokol altında yapılan adil ve kontrollü bir karşılaştırma; ayrıca **donuk-omurga vs ince-ayar ablasyonu**.

> 🇬🇧 For English, see [README.md](README.md).

---

## Motivasyon

Çevrimdışı imza doğrulama çalışmalarının çoğu **tek bir veri seti** (genellikle Latin karakterli CEDAR) ve **tek bir mimari** üzerinde değerlendirilir; raporlanan başarımın diğer yazı sistemlerine genellenip genellenmediği nadiren test edilir. Yöntemsel tercihler çalışmadan çalışmaya değiştiği için, bir başarım farkının **verinin içsel zorluğundan** mı, **mimari seçiminden** mi, yoksa yalnızca yöntem farklılıklarından mı kaynaklandığını anlamak zordur.

Bu proje o belirsizliği ortadan kaldırır. **Her deney tek ve özdeş bir protokolü paylaşır** — aynı hiperparametreler, aynı kod, aynı rastgelelik tohumu. Tek değişkenler: *hangi veri seti*, *hangi senaryo* (yazar-bağımlı/bağımsız), *hangi omurga* ve (ablasyonda) *omurganın donuk mu ince-ayarlı mı* olduğudur.

## Bu sürümde yeni neler var

Çalışma, tek-omurgalı bir temelden tam bir karşılaştırmalı çalışmaya genişletildi:

- **Dört omurga** özdeş protokol altında karşılaştırıldı: ResNet50, VGG19, DenseNet121, EfficientNet-B0.
- En iyi omurga (VGG19) üzerinde **ablasyon çalışması**: donuk omurga vs son-blok ince-ayarı.
- **Biyometrik değerlendirme** (FAR/FRR/AER/EER) yalnızca ana model için değil, **30 modelin tamamı** için hesaplandı.
- **Toplam 30 model**: 4 omurga × 3 veri × 2 senaryo (24) + VGG19 ablasyon × 6.

![Mimari](docs/architecture_multibackbone.png)

## Temel Fikir

- **Donuk omurga** (ImageNet ön-eğitimli): yalnızca sınıflandırıcı kafa eğitilir — veri setleri arası karşılaştırma için adil, düşük-varyanslı bir taban.
- **Veri seti başına iki senaryo:**
  - **Yazar-bağımlı (WD):** rastgele bölme; aynı kişi hem eğitim hem testte olabilir.
  - **Yazar-bağımsız (WI):** kişi-ayrık bölme; test kişileri eğitimde hiç görülmez.
- **Ablasyon:** VGG19'da son evrişim bloğu çözülür (daha küçük lr=1e-5 ile ince-ayar) — donuk-omurga kararının ne kazandırıp ne kaybettirdiğini ölçmek için.
- **Biyometrik değerlendirme:** doğruluk/kesinlik/duyarlılık/F1'in yanı sıra her model için FAR, FRR, AER ve EER raporlanır.

## Sonuçlar — aşama aşama

### Aşama 1 — Başlangıç: tek omurga (ResNet50)

İlk çalışma donuk ResNet50 kullandı. Ana bulguyu — net bir **yazı sistemi zorluk sıralaması, CEDAR < BHSig < UTSig** — ortaya koydu, ancak bu sıralamanın seçilen mimarinin bir yan etkisi olup olmadığı açık kaldı.

![Aşama 1](results/figures/stage1_resnet_eer.png)

### Aşama 2 — Çok-omurga karşılaştırması

Özdeş protokol altında üç omurga daha eklemek, **zorlu Latin-dışı veri setlerinde mimarinin muazzam fark yarattığını** ortaya koyuyor. VGG19, altı senaryonun dördünü kazanıyor ve en önemli yerlerde dramatik biçimde daha iyi: BHSig-WD'de EER'i %24,5'ten (ResNet50) **%4,45'e**, UTSig-WD'de %26,5'ten **%10,27'ye** düşürüyor.

![Aşama 2 — doğruluk](results/figures/backbone_accuracy.png)

![Aşama 2 — EER](results/figures/stage2_backbone_eer.png)

**Sınıflandırma doğruluğu (donuk omurga):**

| Veri / Senaryo | ResNet50 | VGG19 | DenseNet121 | EfficientNet-B0 |
|----------------|---------:|------:|------------:|----------------:|
| CEDAR-WD | 0,945 | **0,977** | 0,962 | 0,970 |
| CEDAR-WI | 0,866 | 0,915 | 0,879 | **0,917** |
| BHSig-WD | 0,772 | **0,958** | 0,799 | 0,754 |
| BHSig-WI | 0,808 | **0,823** | 0,818 | 0,790 |
| UTSig-WD | 0,756 | **0,906** | 0,734 | 0,741 |
| UTSig-WI | 0,728 | 0,663 | **0,735** | 0,701 |

VGG19, en zor yazar-bağımsız durumlar hariç baskın. UTSig-WI'da **aşırı uyum** gösteriyor (duyarlılık 0,33'e düşüyor) ve DenseNet121 öne geçiyor. Yüksek kapasite her zaman en iyi değil.

### Aşama 3 — Ablasyon: donuk vs ince-ayar (VGG19)

Son bloğu çözmek esas olarak **yazar-bağımlı** senaryolarda yardımcı oluyor, yazar-bağımsızda güvenilmez — bu da donuk-omurga seçiminin sağlam bir taban olduğunu doğruluyor.

![Aşama 3 — EER](results/figures/stage3_ablation_eer.png)

| Veri / Senaryo | Donuk EER | İnce-ayar EER | Δ EER |
|----------------|----------:|--------------:|------:|
| CEDAR-WD | 2,46 | **0,76** | −1,70 |
| CEDAR-WI | 8,90 | 8,90 | 0,00 |
| BHSig-WD | 4,45 | **2,28** | −2,17 |
| BHSig-WI | 17,80 | **15,95** | −1,85 |
| UTSig-WD | 10,27 | **7,43** | −2,84 |
| UTSig-WI | 37,73 | 35,10 | −2,63 |

İnce-ayar yazar-bağımlı senaryolarda EER'i düşürüyor (ör. UTSig-WD 10,27→7,43), ama yazar-bağımsız UTSig-WI'da model ciddi aşırı uyum rejiminde kalıyor (FRR ~%60, gerçek imzaların çoğunu reddediyor).

### Ana Bulgu

Net bir **yazı sistemi zorluk sıralaması** dört omurgada da korunuyor: **CEDAR < BHSig < UTSig**. Veri seti etkisi WD–WI senaryo etkisinden baskın ve tek bir veri setinde (veya tek bir mimaride) yüksek doğruluk, tek başına genellenebilirlik kanıtı **değil** — omurga seçimi en zor veri setlerinde sonuçları ~20 EER puanına kadar değiştiriyor.

## Veri Setleri

| Özellik | CEDAR | BHSig | UTSig |
|---------|------:|------:|------:|
| Yazı sistemi | Latin | Devanagari | Perso-Arabik |
| Kişi sayısı | 55 | 260 | 115 |
| Gerçek | 1.320 | 6.240 | 3.105 |
| Sahte | 1.320 | 7.800 | 4.830 |
| Toplam | 2.640 | 14.040 | 7.935 |
| Sahtecilik | Amatör | Yetenekli | Yetenekli |

> Veri setleri burada **paylaşılmaz**. Orijinal kaynaklarından edinip her birini `train/{genuine,forged}` ve `test/{genuine,forged}` biçiminde düzenleyin. Yolları `src/signature_data.py` başından güncelleyin.

## Depo Yapısı

```
.
├── src/
│   ├── signature_data.py            # veri yükleme + WD/WI bölme (üç veri seti)
│   ├── train_unified.py             # eğitim: --dataset --scenario --backbone --finetune
│   ├── compute_metrics_unified.py   # ilk ResNet50 koşusu için biyometrik metrikler
│   ├── compute_biometrics_all.py    # TÜM modeller için FAR/FRR/EER
│   └── aggregate_results.py         # omurga + ablasyon karşılaştırma tabloları/grafikleri
├── results/
│   ├── figures/                     # aşamalı karşılaştırma grafikleri (doğruluk + EER)
│   └── metrics/                     # all_results_classification.csv, all_results_biometric.csv, ablation_vgg19.csv
├── docs/
│   └── architecture_multibackbone.png   # mimari + ablasyon diyagramı
├── paper/                           # makale (docx + pdf)
├── requirements.txt
├── LICENSE
└── README.md
```

## Kurulum

```bash
pip install -r requirements.txt
```

Python 3.9+ ve PyTorch gerektirir. GPU önerilir ama zorunlu değildir (deneyler CPU'da çalıştırılmıştır).

## Kullanım

`src/signature_data.py` başındaki veri yollarını düzenleyin, sonra deneyleri çalıştırın. Her deney veri seti, senaryo, omurga ve ince-ayar moduyla parametrelenir:

```bash
cd src
# Çok-omurga karşılaştırması (donuk omurga)
python train_unified.py --dataset cedar --scenario wd --backbone vgg19
python train_unified.py --dataset bhsig --scenario wd --backbone densenet121
python train_unified.py --dataset utsig --scenario wi --backbone efficientnet_b0
# ... (4 omurga × 3 veri × 2 senaryo)

# Ablasyon (donuk vs ince-ayar, VGG19'da)
python train_unified.py --dataset utsig --scenario wd --backbone vgg19 --finetune last_block
```

Sonra sonuçları toplayın ve tüm modeller için biyometrik metrikleri hesaplayın:

```bash
python aggregate_results.py          # karşılaştırma tabloları + grafikler
python compute_biometrics_all.py     # 30 modelin tamamı için FAR/FRR/AER/EER
```

## Eğitim Protokolü (tüm deneylerde özdeş)

| Hiperparametre | Değer |
|----------------|-------|
| Omurgalar | ResNet50 · VGG19 · DenseNet121 · EfficientNet-B0 (ImageNet, donuk) |
| İnce-ayar (ablasyon) | son evrişim bloğu çözülür, lr=1e-5 |
| Optimizasyon | Adam, lr = 1e-4 (kafa) |
| Ağırlık sönümü | 5e-4 |
| Batch boyutu | 32 |
| Epoch | 30 (erken durdurma, sabır = 5) |
| Dropout | 0,3 |
| Görüntü boyutu | 224 × 224 |
| Tohum | 42 |

## Tekrar Üretilebilirlik

Bölmeler sabit tohumla (42) bellekte yapılır ve tamamen deterministiktir — tüm scriptler aynı `signature_data` modülünü kullandığından, her model tam olarak eğitildiği test kümesinde değerlendirilir. `compute_biometrics_all.py` her kayıtlı modeli yeniden yükleyip aynı deterministik test kümesinde yeniden skorlar.

## Atıf

Bu kodu kullanırsanız ekteki makaleye atıf yapınız (bkz. `paper/`). Yayımlandığında BibTeX kaydı eklenecektir.

## Lisans

MIT Lisansı ile yayımlanmıştır — bkz. [LICENSE](LICENSE).
