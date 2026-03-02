# Project Context — Real-time Generative AI and the Risk of Addiction

## Overview

MSc Artificial Intelligence coursework for **Deep Learning and Generative AI** (CMP030L043, Level 7) at University of Roehampton. Two-part assessment based on **Topic 28** from the suggested paper list.

---

## The Paper

**Title:** Real-Time Hyper-Personalized Generative AI Should Be Regulated to Prevent the Rise of "Digital Heroin"
**Authors:** Raad Khraishi, Cristovao Iglesias de Oliveira, Devesh Batra, Peter Gostev, Giulio Pelosio, Ramin Okhrati, Greig Cowan
**Venue:** NeurIPS 2025 Oral — Position Paper
**Link:** https://neurips.cc/virtual/2025/oral/126308

### Paper's Core Argument
- Real-time generative AI shortens the content-generation feedback loop to seconds, enabling hyper-personalized outputs on the fly.
- When paired with engagement-maximizing incentives, this creates "digital heroin" — unprecedented compulsive consumption patterns.
- Draws on dopamine-driven feedback neuroscience and clinical social media addiction observations.
- Calls for government oversight akin to addictive substance regulation, especially for minors.

### Critical Gap (Our Angle)
**The paper is purely argumentative — it contains NO model, NO dataset, NO experiment, NO metric.** Our proposal fills this empirical void with a testable deep learning pipeline.

---

## Assessment Structure

| Component | Weight | Word Limit | Deadline | Format |
|-----------|--------|------------|----------|--------|
| **Part 1:** Critical Appraisal & Proposal | 40% | 2,000 words | 6 March 2026, 16:00 | PDF via Turnitin |
| **Part 2:** Project (Artefact + Report) | 60% | 3,000 words | 17 April 2026, 16:00 | PDF + GitHub/Colab + Zipped code |

Must achieve **50% in BOTH** components to pass.

### Part 1 Required Sections
1. **Summary** (~200 words) — paper's core contribution, architecture, key results
2. **Critical Appraisal** (~800 words) — theoretical foundations, methodology critique, limitations
3. **Proposal for Improvement** (~1,000 words) — specific technical extension, justification, hypothesized outcome

### Part 2 Required Sections
- Methodology (architecture, preprocessing, training strategy)
- Results & Evaluation (quantitative metrics + qualitative analysis vs. baseline)
- Critical Discussion (challenges, constraints, hypothesis validation)
- Ethics & Scalability

### Referencing
- **Part 1:** Harvard style
- **Part 2:** IEEE style
- Report format: A4, 11-12pt font, single/1.15 spacing, 8-10 pages

---

## Marking Rubric Summary

### Part 1 (40% of module)
| Criteria | Weight | Distinction Target |
|----------|--------|--------------------|
| Critical Analysis | 40% | Deconstruct methodology, identify subtle flaws, link to SOTA |
| Proposed Improvement | 30% | Novel + feasible, technically creative (new loss/architecture), mathematically justified |
| Theoretical Knowledge | 20% | Mastery of DL theory, precise terminology |
| Academic Standards | 10% | Professional narrative, perfect referencing, word count adherence |

### Part 2 (60% of module)
| Criteria | Weight | Distinction Target |
|----------|--------|--------------------|
| Technical Artefact | 40% | Optimized, modular, reproducible code; comprehensive README |
| Evaluation & Results | 30% | Advanced metrics (FID, Perplexity), ablation studies, error bars, critical analysis |
| Critical Reflection | 20% | Deep discussion of challenges, theory-practice gap |
| Ethics & Scalability | 10% | Specific ethical risks + mitigations, computational cost analysis |

---

## Our Proposed Solution: DT-AttNet

### Dual-Head Temporal Attention Network

Converts the paper's qualitative "shortened dopamine loop" thesis into a **quantifiable addiction risk scoring model**.

### Architecture
```
Input (B, T, F) → 1D-CNN (k=3, 64 filters) → BatchNorm → ReLU
    → BiLSTM (hidden=128) → (B, T, 256)
    → Multi-Head Self-Attention (4 heads, d=256) → Residual + LayerNorm
    → Global Average Pool → (B, 256)
    ├→ Classifier Head: Linear→ReLU→Dropout(0.3)→Linear→Softmax → 3 classes
    └→ Projection Head: Linear→ReLU→Linear → 64-dim (for SupCon loss)
```

### Loss Function
`L = CrossEntropy(logits, y) + λ × SupConLoss(projections, y)`
- λ = 0.5 (tunable), temperature = 0.07
- SupCon (Khosla et al., 2020) improves class separation for imbalanced addiction levels

### Output Classes
- **Low** — healthy engagement patterns
- **Moderate** — at-risk engagement patterns
- **High** — addiction-indicative engagement patterns

---

## Datasets

| Dataset | Source | Records | License | Features |
|---------|--------|---------|---------|----------|
| Social Media Addiction & Usage Patterns | Kaggle | ~1,000+ | CC0 | daily_usage, platform, night_usage, mental_health, productivity, addiction_level |
| Smartphone Usage & Addiction Prediction | Kaggle | ~7,500 | CC0 | screen_time, social_media, gaming, sleep, stress, addiction_severity |

**Constraint:** All training must run on **free-tier GPU** (Kaggle T4 or Google Colab T4).
Model is ~500K parameters — trains in minutes on T4.

---

## Experiments Design

| ID | Model | Loss | Purpose |
|----|-------|------|---------|
| E1 | Baseline MLP | CE | Baseline comparison (no temporal modeling) |
| E2 | DT-AttNet | CE only | Ablation: is contrastive loss needed? |
| E3 | DT-AttNet | CE + SupCon | **Full proposed model** |
| E4 | DT-AttNet (no attention) | CE + SupCon | Ablation: is attention needed? |
| E5 | DT-AttNet (λ sweep) | CE + SupCon (λ=0.1,0.5,1.0) | Hyperparameter sensitivity |

### Metrics
- Primary: Accuracy, Macro-F1, Weighted-F1
- Per-class: Precision, Recall, F1
- Advanced: ROC-AUC (one-vs-rest), attention weight statistics

### Hypotheses
- **H1:** DT-AttNet outperforms baseline MLP on accuracy and macro-F1
- **H2:** Attention weights reveal re-engagement interval shortening as strongest predictor (validates paper's thesis)
- **H3:** Contrastive loss improves Moderate vs. High class separation

---

## Tech Stack

- **Language:** Python 3.10+
- **Framework:** PyTorch 2.x
- **Environment:** Google Colab / Kaggle Notebooks (free T4 GPU)
- **Key Libraries:** scikit-learn, pandas, matplotlib, seaborn, PyYAML
- **Version Control:** GitHub (private until submission)

---

## Repository Structure

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
│   ├── model.py           # DTAttNet + BaselineMLP
│   ├── losses.py           # CombinedLoss (CE + SupCon)
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
└── LICENSE
```

---

## Key References

| # | Reference | Context |
|---|-----------|---------|
| 1 | Khraishi et al. (2025) — NeurIPS | Target paper |
| 2 | Khosla et al. (2020) — NeurIPS | Supervised Contrastive Loss |
| 3 | Vaswani et al. (2017) | Self-attention mechanism |
| 4 | Hochreiter & Schmidhuber (1997) | LSTM backbone |
| 5 | Montag et al. (2019) | Dopamine and social media |
| 6 | Sherman et al. (2016) | Neural activation from notifications |
| 7 | Haidt (2024) — *The Anxious Generation* | Youth + tech harm |
| 8 | Lembke (2021) — *Dopamine Nation* | Clinical dopamine dysregulation |
| 9 | Alter (2017) — *Irresistible* | Behavioral addiction |
| 10 | Sutton & Barto (2018) | RL theory (engagement optimization framing) |
| 11 | Ismail Fawaz et al. (2019) | Deep learning for time-series |
| 12 | EU AI Act (2024) | Regulatory framework |
| 13 | Goodfellow et al. (2016) | Deep Learning textbook |

---

## Guide Documents

- [Part1_Report_Guide.md](file:///c:/Users/VAMSHI%20KRISHNA/OneDrive/Documents/deep%20leraning%20course%20work/Part1_Report_Guide.md) — Section-by-section writing guide for Part 1 report
- [Part2_Implementation_Guide.md](file:///c:/Users/VAMSHI%20KRISHNA/OneDrive/Documents/deep%20leraning%20course%20work/Part2_Implementation_Guide.md) — Step-by-step implementation guide with full PyTorch code

---

## Important Constraints

- **No AI-written report text** — all final writing must be the student's own
- **AI code assistance must be acknowledged** in submission
- **Free GPU only** — Colab/Kaggle T4, no paid compute
- **Reproducibility required** — notebook must run top-to-bottom with outputs visible
- **Private repo** until after grading
