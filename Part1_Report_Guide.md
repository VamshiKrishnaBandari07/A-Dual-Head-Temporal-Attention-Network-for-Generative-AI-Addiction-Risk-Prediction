# Part 1: Critical Appraisal & Proposal — Writing Guide
## Topic 28: Real-time Generative AI and the Risk of Addiction ("Digital Heroin")

> **Paper:** Khraishi, R., Iglesias de Oliveira, C., Batra, D., Gostev, P., Pelosio, G., Okhrati, R. & Cowan, G. (2025). *Real-Time Hyper-Personalized Generative AI Should Be Regulated to Prevent the Rise of "Digital Heroin."* NeurIPS 2025 Oral (Position Paper).
>
> **Constraints:** ~2,000 words | 40% of module mark | Harvard referencing | PDF via Turnitin

---

## Multi-Turn Self-Critique of the Approach

Before defining the report structure, the approach was refined through iterative self-interrogation:

### Round 1 — Initial Proposal
> **Idea:** Simply summarize the paper's addiction argument and propose "adding a content filter" as improvement.
>
> **Self-Critique:** ❌ Too superficial. The rubric demands *technical creativity* (e.g., new loss term, architectural block). A "content filter" is vague, is not a DL contribution, and shows no understanding of the paper's actual gap: *it is qualitative/argumentative with NO computational model*. The biggest critique must be: **the paper proposes no formal model, no dataset, no metric, no experiment.** A strong proposal must fill that void.

### Round 2 — Refined Proposal
> **Idea:** Propose an *Addiction Risk Scoring Model* — a deep learning classifier that takes user behavioral features (session time, scroll speed, re-engagement intervals) and predicts addiction severity. This reframes the paper's qualitative "dopamine loop" narrative into a *quantifiable, testable DL pipeline.*
>
> **Self-Critique:** ✅ Better — it's technical and feasible. But is it novel enough? The rubric says "novel and feasible... demonstrates technical creativity." A plain MLP classifier on tabular data is competent but not distinction-level.

### Round 3 — Final Proposal (Distinction-Level)
> **Idea:** A **Dual-Head Temporal Attention Network (DT-AttNet)** — a custom PyTorch architecture with:
> 1. **Temporal Engagement Encoder** (1D-CNN + BiLSTM) to model sequential user interaction patterns over time
> 2. **Self-Attention Risk Head** to weight which behavioral moments most strongly predict addiction escalation
> 3. **Contrastive Regularization Loss** to better separate "at-risk" vs "healthy" engagement in the latent space
>
> This turns the paper's theoretical "shortening feedback loop" claim into a measurable, predictive experiment.
>
> **Self-Critique:** ✅ This is technically creative (custom architecture, new loss term), directly addresses the paper's gap (no empirical model), and is feasible on Kaggle/Colab free GPU with tabular + time-sequence data. This is the approach to take.

---

## Report Structure & Section-by-Section Guide

### 1. Summary (~200 words)

**What to write:**
- Paper's core argument: real-time generative AI that hyper-personalises content on-the-fly creates "digital heroin" by shortening dopamine-driven feedback loops to seconds.
- It is a **position paper** (NeurIPS 2025 Oral) — meaning it argues a thesis rather than presenting a new model.
- Key claims: misaligned engagement incentives + real-time personalization = compulsive consumption, especially harmful to adolescents.
- The paper calls for substance-level regulatory oversight and proactive ML community guidelines.

**Key references to include:**
- Khraishi et al. (2025) — the paper itself
- Montag et al. (2019) — dopamine and social media
- Lembke (2021) — *Dopamine Nation* (clinical context)

---

### 2. Critical Appraisal (~800 words)

Structure this into **three clear subsections**:

#### 2.1 Strengths — Theoretical Foundations (~250 words)
Evaluate positively:
- Interdisciplinary scope (neuroscience + ML + public policy) — rarely seen at ML venues.
- Timely argument: positions generative AI as an *acceleration* of existing social media addiction patterns (cite Alter, 2017; Haidt, 2024).
- Sound dopamine cycle framing: references "variable ratio reinforcement" (Skinner, 1957); cite neuroimaging studies showing striatal activation from social media notifications (Sherman et al., 2016).

Link to module content:
- Discuss how engagement optimisation is essentially **reward maximisation** in an RL framework (Sutton & Barto, 2018).
- Explain how recommendation models (deep collaborative filtering, transformers) are trained with **cross-entropy loss on click-through rate** — the objective function itself encodes the addictive incentive.

#### 2.2 Methodological Critique (~300 words)
Critique rigorously:
- **No empirical validation.** The paper is purely argumentative — no dataset, no model, no experiment, no metric. For a venue that values reproducibility, this is a significant limitation.
- **No baseline comparison.** Claims about "shortened loops" are not benchmarked against actual engagement data from legacy (pre-GenAI) platforms vs. GenAI-powered platforms.
- **Missing quantification of the feedback loop.** How short is "mere seconds"? What is the threshold at which personalization speed transitions from useful to harmful? The paper does not operationalize its central claim.
- **Selection bias in cited literature.** The neuroscience citations primarily cover *social media addiction* (Instagram, TikTok), not generative AI specifically. The leap from scroll-based engagement to GenAI-produced content carries differences (user is co-creating, not passively consuming) that the paper does not address.
- **Regulatory proposal lacks technical specificity.** Calls for "government oversight akin to addictive substances" without proposing *what* to measure, *how* to audit, or *who* certifies.

#### 2.3 Limitations & Ethical Risks (~250 words)
Analyse:
- **Surveillance trade-off:** Any addiction monitoring system requires collecting granular behavioral data, creating privacy risks (reference GDPR, AI Act 2024).
- **Computational cost of real-time monitoring** is itself a concern — does always-on monitoring at scale justify the environmental and resource cost?
- **Paternalism risk:** The paper implies user autonomy should be overridden for "protection," but doesn't engage with agency-preserving frameworks (Thaler & Sunstein, 2008 — nudge theory).
- **Generalizability gap:** Adolescent-specific claims cannot be simply extended to all demographics without stratified analysis.

---

### 3. Proposal for Improvement (~1,000 words)

Structure into **four subsections**:

#### 3.1 Motivation from Critique (~200 words)
- The paper's fatal gap: it argues *what* the risk is but provides no framework *how* to detect, measure, or mitigate it computationally.
- Proposal: A **Dual-Head Temporal Attention Network (DT-AttNet)** that converts the paper's qualitative claims into a testable deep learning pipeline.
- Goal: predict user addiction risk level from sequential behavioral interaction data.

#### 3.2 Proposed Architecture (~350 words)
Describe the architecture precisely with technical detail:

1. **Input representation:** Sequential user interaction records — each timestep includes features: *session_duration*, *scroll_speed*, *re-engagement_interval*, *content_type_consumed*, *platform*, *time_of_day*.
2. **Temporal Engagement Encoder:**
   - 1D Convolutional layer (kernel=3, filters=64) to capture local temporal patterns in engagement behavior.
   - Bidirectional LSTM (hidden_dim=128) to model long-range dependencies across sessions.
3. **Self-Attention Risk Head:**
   - Multi-head self-attention (4 heads, d_model=256) over the BiLSTM outputs.
   - Purpose: learn *which temporal windows* most predict addiction escalation — directly operationalizing the paper's "shortening loop" argument.
4. **Classification Head:** Linear → ReLU → Dropout(0.3) → Linear → Softmax over 3 classes: {Low, Moderate, High} addiction risk.
5. **Contrastive Regularization Loss:**
   - Combined loss = CrossEntropyLoss + λ × SupConLoss (Supervised Contrastive Loss, Khosla et al., 2020).
   - Rationale: pulls embeddings from the same addiction class together while pushing different classes apart, improving decision boundary quality beyond standard CE.

Include a **figure/diagram placeholder** showing the architecture flow.

#### 3.3 Justification (~200 words)
- The 1D-CNN + BiLSTM backbone is well-suited for sequential tabular data (cite Ismail Fawaz et al., 2019 on deep learning for time-series).
- Self-attention weights are *interpretable* — they expose which behavioral moments matter most, directly supporting the regulatory audit trail the paper advocates for.
- Contrastive loss has shown improvements in imbalanced classification settings (Khosla et al., 2020), relevant here since extreme addiction is rarer.
- Feasibility: entire pipeline can run on a Kaggle T4 GPU / Colab free tier with the identified datasets.

#### 3.4 Hypothesized Outcome (~250 words)
- **H1:** DT-AttNet will outperform a baseline MLP (accuracy, macro-F1) on addiction severity classification.
- **H2:** Attention weight analysis will reveal that *re-engagement interval shortening* over time is the strongest predictor of high addiction risk — providing computational evidence for the paper's core "shortened loop" thesis.
- **H3:** Contrastive loss will improve separation of Moderate vs. High risk classes (often confused by standard classifiers).
- State explicitly: Part 2 will implement this architecture and test these hypotheses.

---

## Reference Baseline (Minimum ~15-20 References)

| # | Reference | Purpose |
|---|-----------|---------|
| 1 | Khraishi et al. (2025) — NeurIPS | The target paper |
| 2 | Alter, A. (2017). *Irresistible* | Behavioral addiction framework |
| 3 | Lembke, A. (2021). *Dopamine Nation* | Clinical dopamine dysregulation |
| 4 | Montag et al. (2019) — *Addictive Behaviors* | Dopamine and social media |
| 5 | Sherman et al. (2016) — *Psychological Science* | Neural activation from likes/notifications |
| 6 | Haidt, J. (2024). *The Anxious Generation* | Youth + technology harm evidence |
| 7 | Skinner, B.F. (1957) — Variable-ratio reinforcement | Classical conditioning basis |
| 8 | Sutton & Barto (2018). *Reinforcement Learning* (2nd ed.) | RL theory for engagement optimization |
| 9 | Thaler & Sunstein (2008). *Nudge* | Choice architecture / paternalism |
| 10 | Khosla et al. (2020) — *NeurIPS* | Supervised Contrastive Learning |
| 11 | Ismail Fawaz et al. (2019) — *Data Mining & Knowledge Discovery* | Deep learning for time-series classification |
| 12 | EU AI Act (2024) | Regulatory framework reference |
| 13 | WHO (2018) — ICD-11 Gaming Disorder | Clinical classification of digital addiction |
| 14 | Covington et al. (2016) — YouTube RecSys | Deep neural net recommendation systems |
| 15 | Vaswani et al. (2017) — "Attention Is All You Need" | Self-attention mechanism |
| 16 | Hochreiter & Schmidhuber (1997) — LSTM | Sequence modeling backbone |
| 17 | Goodfellow et al. (2016) — *Deep Learning* textbook | Theoretical foundations |

---

## Formatting Checklist

- [ ] Harvard referencing throughout (Author, Year) — in-text and full reference list
- [ ] 2,000 words (±10%)
- [ ] PDF format via Turnitin
- [ ] Clear subsection headings: 1. Summary, 2. Critical Appraisal, 3. Proposal
- [ ] Formal academic tone — no first-person, no colloquialisms
- [ ] Architecture diagram for the proposal section
- [ ] Acknowledge any AI tool assistance at the end

---

## Key Pitfalls to Avoid

> [!CAUTION]
> - **Do NOT just summarize the paper.** The rubric penalizes "descriptive only" analysis. Every paragraph in critical appraisal must evaluate, not describe.
> - **Do NOT propose "add more data" or "add more layers."** The rubric calls this "standard extension without deep justification" (Pass band, NOT distinction).
> - **Do NOT ignore feasibility.** The proposal must be implementable in Part 2 with free GPU resources.
> - **Do NOT miss the position paper angle.** This paper is NOT a standard ML paper — it has no model. Your critique must acknowledge this and exploit it as the primary gap.
