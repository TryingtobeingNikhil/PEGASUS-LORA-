# PEGASUS + LoRA: Parameter-Efficient Text Summarization

An empirical research implementation comparing Full-Parameter Fine-Tuning vs. Low-Rank Adaptation (LoRA) for abstractive text summarization using Google's PEGASUS architecture.

## 📌 Project Overview
Large Language Models (LLMs) like PEGASUS (568M+ parameters) are computationally expensive to fine-tune. This project demonstrates how to implement **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA** to drastically reduce training time and VRAM requirements while mitigating catastrophic forgetting. 

The experiment uses `google/pegasus-xsum` to test how the model adapts to out-of-domain text (tech product descriptions, local government policy) when given extremely small datasets.

## 🛠️ Tech Stack
* **Framework:** PyTorch
* **Architecture:** Transformer (Encoder-Decoder)
* **Libraries:** Hugging Face `transformers`, `peft` (LoRA), `evaluate` (ROUGE scores)
* **Base Model:** `google/pegasus-xsum`

## 📊 Empirical Comparison: Full Fine-Tuning vs. LoRA
We trained the model on a micro-dataset to observe overfitting and computational overhead.

| Metric | Full Fine-Tuning | LoRA Fine-Tuning | Reduction / Improvement |
| :--- | :--- | :--- | :--- |
| **Trainable Parameters** | 767,616,000 (100%) | 1,572,864 (0.20%) | **99.8% Reduction** |
| **Training Time (5 Epochs)** | ~260 seconds | ~9.5 seconds | **27x Faster** |
| **Final Training Loss** | 3.77 | 3.45 | **Improved Convergence** |

## 🧠 Key Findings & Hallucination Analysis

### Root Cause of Hallucinations: Domain Mismatch
During testing, the model hallucinated specific entities (e.g., inventing a 2017 "Samsung Galaxy S8" or hallucinating "Adelaide, Australia" for generic city council text). 
* **The Cause:** `pegasus-xsum` is heavily pre-trained on BBC news (UK-centric). When presented with generic text in a zero-shot or under-trained scenario, the model's self-attention mechanism falls back on high-probability BBC pre-training biases.
* **The LoRA Advantage:** While full fine-tuning resulted in severe "word salad" (catastrophic forgetting), LoRA successfully preserved the model's syntactic fluency while reducing the training footprint by 99.8%.

### How to Scale for Production
To completely eliminate domain-mismatch hallucinations:
1. Increase the dataset from a micro-batch (3 examples) to 100+ domain-specific examples.
2. Utilize domain-matched base models (e.g., `pegasus-cnn_dailymail` for general press releases).
3. Deploy the resulting lightweight LoRA adapters (~6MB) as swappable modules for different summarization tasks.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/TryingtobeingNikhil/pegasus-lora-summarization.git](https://github.com/TryingtobeingNikhil/pegasus-lora-summarization.git)
