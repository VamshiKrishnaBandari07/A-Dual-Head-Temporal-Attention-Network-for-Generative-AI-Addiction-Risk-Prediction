# DT-AttNet: Dual-Head Temporal Attention Network for Addiction Risk Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform: Colab/Kaggle](https://img.shields.io/badge/GPU-T4%20Free%20Tier-orange.svg)]()

> A deep learning framework that converts the qualitative "digital heroin" thesis from Khraishi *et al.* (NeurIPS 2025) into a quantifiable addiction risk scoring pipeline using temporal behavioural modelling.

---

## Overview

This project is part of the MSc Artificial Intelligence coursework for **Deep Learning and Generative AI** (CMP030L043, Level 7) at the University of Roehampton. It addresses **Topic 28** — *Real-Time Hyper-Personalized Generative AI and the Risk of Addiction*.

The target paper argues that real-time generative AI creates "digital heroin" through shortened dopamine-driven feedback loops, but provides **no computational model, no dataset, and no experiment**. This project fills that empirical void with **DT-AttNet** — a custom PyTorch architecture that predicts user addiction risk level from sequential behavioural interaction data.

---

## Architecture

```
Input (B, T, F) → 1D-CNN (k=3, 64 filters) → BatchNorm → ReLU
    → BiLSTM (hidden=128) → (B, T, 256)
    → Multi-Head Self-Attention (4 heads, d=256) → Residual + LayerNorm
    → Global Average Pool → (B, 256)
    ├→ Classification Head → Softmax → {Low, Moderate, High}
    └→ Projection Head → 64-dim → SupCon Loss
```

**Loss Function:** `L = CrossEntropy(logits, y) + λ × SupConLoss(projections, y)` where λ = 0.5, τ = 0.07

**Parameters:** ~500K (trains in minutes on a T4 GPU)

---

## Assessment Structure

| Component | Weight | Word Limit | Deadline |
|-----------|--------|------------|----------|
| **Part 1:** Critical Appraisal & Proposal | 40% | 2,000 words | 6 March 2026 |
| **Part 2:** Implementation + Report | 60% | 3,000 words | 17 April 2026 |

---

## Datasets

| Dataset | Records | License | Source |
|---------|---------|---------|--------|
| Social Media Addiction & Usage Patterns | ~1,000+ | CC0 | [Kaggle](https://www.kaggle.com/) |
| Smartphone Usage & Addiction Prediction | ~7,500 | CC0 | [Kaggle](https://www.kaggle.com/) |

---

## Experiments

| ID | Model | Loss | Purpose |
|----|-------|------|---------|
| E1 | Baseline MLP | CE | Baseline (no temporal modelling) |
| E2 | DT-AttNet | CE only | Ablation: contrastive loss contribution |
| E3 | DT-AttNet | CE + SupCon | **Full proposed model** |
| E4 | DT-AttNet (no attention) | CE + SupCon | Ablation: attention contribution |
| E5 | DT-AttNet (λ sweep) | CE + SupCon | Hyperparameter sensitivity |

### Hypotheses

- **H1:** DT-AttNet outperforms baseline MLP on accuracy and macro-F1
- **H2:** Attention weights reveal re-engagement interval shortening as the strongest addiction predictor
- **H3:** Contrastive loss improves Moderate vs. High class separation

### Metrics

Accuracy · Macro-F1 · Weighted-F1 · Per-class Precision/Recall · ROC-AUC (one-vs-rest) · Attention weight analysis

---

## Project Structure

```
addiction-risk-dtattnet/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── download_data.py
│   └── README.md
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Training.ipynb
│   └── 03_Evaluation.ipynb
├── src/
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── model.py              # DT-AttNet + BaselineMLP
│   ├── losses.py              # CombinedLoss (CE + SupCon)
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── experiments/
│   ├── baseline_mlp/
│   └── ablation/
├── results/
│   ├── figures/
│   ├── attention_maps/
│   └── metrics.json
├── figures/                   # Report diagrams
│   ├── fig1_dtattnet_architecture.png
│   └── fig2_experimental_pipeline.png
├── Part1_Report.md
├── PROJECT_CONTEXT.md
├── Part1_Report_Guide.md
└── LICENSE
```

---

## Tech Stack

- **Language:** Python 3.10+
- **Framework:** PyTorch 2.x
- **Environment:** Google Colab / Kaggle Notebooks (free T4 GPU)
- **Libraries:** scikit-learn, pandas, matplotlib, seaborn, PyYAML
- **Version Control:** GitHub (private until submission)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/<username>/addiction-risk-dtattnet.git
cd addiction-risk-dtattnet

# Install dependencies
pip install -r requirements.txt

# Download datasets
python data/download_data.py

# Run training (full model)
python src/train.py --config config.yaml --experiment E3
```

Or open the notebooks directly in [Google Colab](https://colab.research.google.com/) or [Kaggle](https://www.kaggle.com/).

---

## References

1. Khraishi *et al.* (2025) — "Digital Heroin" — NeurIPS 2025 Oral
2. Khosla *et al.* (2020) — Supervised Contrastive Learning — NeurIPS
3. Vaswani *et al.* (2017) — Attention Is All You Need
4. Hochreiter & Schmidhuber (1997) — LSTM
5. Ismail Fawaz *et al.* (2019) — Deep Learning for Time-Series Classification

---

## Acknowledgements

This project is submitted as coursework for the University of Roehampton MSc AI programme. AI assistance was used for structural planning and diagram generation; all analytical content and implementation is the author's own work.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
