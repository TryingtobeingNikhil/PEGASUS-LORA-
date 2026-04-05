# PEGASUS + LoRA: Parameter-Efficient Text Summarization

An empirical research implementation comparing **Full-Parameter Fine-Tuning** vs. **Low-Rank Adaptation (LoRA)** for abstractive text summarization using Google's PEGASUS architecture.

---

## 📌 Project Overview
Large Language Models (LLMs) like **PEGASUS (568M+ parameters)** are computationally expensive to fine-tune. This project demonstrates how to implement **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA** to drastically reduce training time and VRAM requirements while mitigating catastrophic forgetting.

The experiment uses `google/pegasus-xsum` to test how the model adapts to **out-of-domain text** (tech product descriptions, local government policy) when given extremely small datasets.

---

## 🛠️ Tech Stack
- **Framework:** PyTorch  
- **Architecture:** Transformer (Encoder–Decoder)  
- **Libraries:** Hugging Face `transformers`, `peft` (LoRA), `evaluate` (ROUGE metrics)  
- **Base Model:** `google/pegasus-xsum`

---

## 📊 Empirical Comparison: Full Fine-Tuning vs. LoRA

We trained the model on a **micro-dataset** to highlight differences in training efficiency and overfitting behavior.

| Metric | Full Fine-Tuning | LoRA Fine-Tuning | Reduction / Improvement |
|------|------|------|------|
| **Trainable Parameters** | 767,616,000 (100%) | 1,572,864 (0.20%) | **99.8% Reduction** |
| **Training Time (5 Epochs)** | ~260 seconds | ~9.5 seconds | **27× Faster** |
| **Final Training Loss** | 3.77 | 3.45 | **Improved Convergence** |

---

##  Key Findings & Hallucination Analysis

### Root Cause of Hallucinations: Domain Mismatch

During testing, the model hallucinated specific entities (e.g., inventing a *2017 Samsung Galaxy S8* or hallucinating *Adelaide, Australia* for generic city council text).

**Cause:**  
`pegasus-xsum` is heavily fine-tuned on **BBC news articles**, making it biased toward journalistic patterns. When exposed to unfamiliar domains with minimal training data, the model's attention layers fall back on **high-probability patterns learned during pretraining**.

**LoRA Advantage:**  
While full fine-tuning caused **catastrophic forgetting** and incoherent outputs ("word salad"), LoRA preserved the model’s linguistic structure by updating only small low-rank matrices.

Result:
- Stable summaries
- 99.8% fewer trainable parameters
- Significantly faster training.

---

## How to Scale for Production

To mitigate hallucinations and improve domain adaptation:

1. Increase dataset size from **3 samples → 100+ domain-specific examples**
2. Use **domain-aligned base models** (e.g., `pegasus-cnn_dailymail`)
3. Deploy **LoRA adapters (~6MB)** as modular components for different domains.

