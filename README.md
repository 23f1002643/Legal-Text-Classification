<div align="center">

# ⚖️ Legal & Policy Document Classification 📜
### 🧠 Transformer Fine-Tuning · LoRA (PEFT) vs Full Fine-Tuning

**Deep Learning Practice (DLP) · NPPE-1 · IIT Madras BS Degree Program**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=for-the-badge&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-8A2BE2?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Task](https://img.shields.io/badge/Task-Text_Classification-success?style=for-the-badge)
![Classes](https://img.shields.io/badge/Classes-30-orange?style=for-the-badge)
![Metric](https://img.shields.io/badge/Metric-Accuracy-blue?style=for-the-badge)

</div>

---

## 🎯 Problem Statement

Build a **Transformer-based text classifier** that automatically sorts **EU legal & policy documents** (Council Decisions, Commission Regulations, Directives, …) into **30 categories**. 🏛️

Unlike short-text tasks, these documents are **long, information-rich and vary a lot in length and style** 📏 — so preprocessing, truncation, model choice and hyperparameters all matter.

### 🔑 What this challenge focuses on

- 🧹 Efficient **preprocessing & tokenization** pipeline
- 🤖 Fine-tuning **pretrained encoder models** for sequence classification
- ⚡ **Parameter-Efficient Fine-Tuning** (LoRA / QLoRA)
- 🎛️ Tuning **learning rate, batch size & training schedule**
- 🔬 **Comparing architectures** and analysing design choices
- 🌍 Generalising to **unseen documents**

---

## 📦 Dataset

| 📁 File | 🧾 Columns | 📝 Notes |
|---|---|---|
| `train.csv` | `id`, `text`, `label` | ~50.8K docs, **30 classes**, long-tailed distribution 📉 |
| `test.csv` | `id`, `text` | Unseen documents to predict |
| `sample_submission.csv` | `ID`, `label` | Required submission format |

📏 **Metric:** Accuracy Score ✅

---

## 📓 Notebooks in this Repo

| 📔 Notebook | 🧪 Method | 🤖 Base Model | 🏆 Result |
|---|---|---|---|
| `dlp-nppe1-lora-roberta-base.ipynb` | ⚡ **LoRA (PEFT)** | `roberta-base` | **0.708** (Leaderboard) |
| `dlp-nppe1-fft-legal-bert.ipynb` | 🔥 **Full Fine-Tuning** | `nlpaueb/legal-bert-base-uncased` | **~0.73 – 0.75** ⭐ |

---

## ⚔️ LoRA vs Full Fine-Tuning

| ⚙️ Aspect | ⚡ LoRA / PEFT | 🔥 Full Fine-Tuning |
|---|---|---|
| 🤖 Base model | `roberta-base` | `legal-bert-base-uncased` |
| 🎚️ Trainable params | ~1.2M (**~1%**) | ~110M (**100%**) |
| 📈 Learning rate | `2e-4` (adapters need higher LR) | `2e-5` (gentle LR) |
| 🔁 Epochs | 4 | up to 6 (early stopping) |
| 💾 Saved artifact | Tiny adapter (few MB) 🪶 | Full checkpoint (~500MB) 🐘 |
| ⏱️ Training time (T4) | ~1.5 hrs | ~1.5 – 2 hrs |
| 🏆 Accuracy | 0.708 | ~0.73 – 0.75 |

> 💡 **Takeaway:** LoRA is cheaper to train, store and swap — but a **domain-matched model (Legal-BERT) with full fine-tuning** had more headroom on this task.

---

## 🛠️ Pipeline

```text
📄 Raw Text
   │
   ▼
🧹 Light Cleaning        → collapse whitespace only (keep legal cues like "Regulation (EC) No.")
   │
   ▼
✂️ Stratified Split       → 90% train / 10% val (all 30 classes preserved)
   │
   ▼
🔤 Tokenization          → truncation @ MAX_LEN = 384, dynamic padding
   │
   ▼
🧠 Model                 → LoRA (roberta-base)  |  Full FT (legal-bert)
   │
   ▼
🚀 Training              → fp16 · gradient checkpointing · group_by_length · eff. batch = 32
   │
   ▼
📊 Evaluation            → Accuracy on validation set
   │
   ▼
📤 submission.csv        → ID, label
```

### ⚡ LoRA configuration
| Param | Value |
|---|---|
| `r` (rank) | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.1 |
| `target_modules` | `query`, `value` |

### 🔥 Full fine-tuning configuration
| Param | Value |
|---|---|
| Learning rate | 2e-5 |
| Warmup ratio | 0.06 |
| Weight decay | 0.01 |
| Batch | 8 × 4 accumulation = **32** |
| Callbacks | ⏹️ EarlyStopping (patience 2) |

---

## 🧪 Experiments Log

| # | 🔬 Approach | 📊 Result | 📝 Verdict |
|---|---|---|---|
| 1 | LoRA + `roberta-base` (r=16) | **0.708** | ✅ Fast & light, but capped |
| 2 | LoRA + `deberta-v3-base` + heavy class-weights + high LR | ~0.31 (epoch 1) | ❌ Unstable, abandoned |
| 3 | **FFT + `legal-bert-base-uncased`** | **~0.73–0.75** | 🏆 Best & most stable |
| 4 | FFT + more dropout + class-weights + long warmup | Regressed | ❌ Over-regularised, reverted |

---

## 💡 Key Learnings

- 🎯 **Domain-matched pretraining wins** — Legal-BERT (trained on legal text) beat general-purpose RoBERTa.
- 🧹 **Less cleaning is more** — legal formatting tokens carry real signal.
- 📉 **LoRA ≠ full FT learning rate** — reusing `2e-4` on full fine-tuning destabilises training.
- ⚖️ **Over-regularisation hurts** — extra dropout + class-weights under-fit the model.
- 🐢 **Long-tailed labels** — rare classes are the hardest and sometimes never predicted.

---

## 🔮 Future Work

- ✂️ **Head + tail truncation** to keep the ending of long documents
- 🤝 **Ensembling** LoRA-RoBERTa + FFT Legal-BERT (average probabilities)
- 📚 Chunk-and-aggregate strategy for very long documents
- 🧠 Try other encoders (DeBERTa-v3, Longformer) with tuned LR

---

## 🚀 How to Run

1. 📥 Open the notebook on **Kaggle** (or upload it there)
2. ➕ Add the competition data: `dlp-nppe-1-t-22026`
3. ⚙️ Settings → Accelerator → **GPU (T4 / P100)** 🎮
4. ▶️ **Run All** — outputs `submission.csv` in `/kaggle/working/`

```bash
pip install transformers==4.41.2 accelerate==0.30.1 datasets==2.19.1
pip install peft==0.11.1   # only for the LoRA notebook
```

---

## 🗂️ Repository Structure

```text
📦 DLP-NPPE1-Legal-Text-Classification
 ┣ 📔 dlp-nppe1-lora-roberta-base.ipynb
 ┣ 📔 dlp-nppe1-fft-legal-bert.ipynb
 ┗ 📄 README.md
```

---

## 🔗 Other DLP NPPE Repos

| 📝 Exam | 🧩 Modality | 📂 Repo |
|---|---|---|
| **NPPE-1** | 📜 Text | *(this repo)* |
| **NPPE-2** | 🎙️ Speech | _coming soon_ |
| **NPPE-3** | 🖼️ Images | _coming soon_ |

---

<div align="center">

### 👨‍💻 Author
**Saini** · IIT Madras BS Degree Program 🎓

⭐ If this helped you, drop a star on the repo! ⭐

*Made with ❤️, ☕ and a lot of GPU hours* 🔥

</div>
