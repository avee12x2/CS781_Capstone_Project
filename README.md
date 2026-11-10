# CS781 Capstone — In-House Legal Intelligence System

**Aarav Oswal (230012), IIT Kanpur — CS781: Large Language Models, Nov 2026**

A confidential, self-hostable legal NLP stack built **from scratch**: our own 32k byte-level BPE tokenizer and five ≈110M-parameter Transformer language models pretrained on the official Indian legal corpus (Phase 1). Six task systems are derived from them (Phases 2–7) using full fine-tuning, partial fine-tuning, LoRA / rsLoRA / LoRMA, adapters, IA³, prompt tuning, BitFit and knowledge distillation. No pretrained model weights, external corpora or LLM APIs are used anywhere in the models.

| Phase | Task | Best test submission | Official primary metric |
|---|---|---|---|
| 1 | Legal LM pretraining (128-word continuation) | A0 GPT-2, SC80 mix, 1B tokens | BERTScore-F1 **0.8393** |
| 2 | Legal NER (12 entity types, strict spans) | A1 + BiLSTM, full fine-tuning | strict macro-F1 **0.6155** |
| 3 | Rhetorical role prediction (13 roles) | F + IA³, separator pooling | macro-F1 **0.5749** |
| 4 | Court judgment prediction (ILDC) | E, 4-chunk encoder → 16-chunk hierarchical BiGRU | macro-F1 **0.796** |
| 5 | Legal statute identification (100 statutes) | E, 8 × 512 chunks, label-wise attention, ASL | macro-F1 **0.4107** |
| 6 | Abstractive summarization | A1 + LoRA, chunk-level seq2seq | BERTScore-F1 **0.8509** |
| 7 | Prior case retrieval | E + LoRMA bi-encoder + lexical hybrid | Micro-F1@7 **0.4761** |

Every phase has ≥ 4 distinct submissions from different Phase-1 backbones; LoRMA and knowledge distillation were used in every phase from 2 to 7. The full write-up is in the project report `CS781-CP-Aarav-Oswal-230012-report.pdf` (submitted on Hello IITK). Codabench username: `a_230012`.

## Repository layout

```
phase1/   tokenizer, corpus encoding, pretraining of A0/A1/B/E/F, decoding, run logs
phase2/   L-NER (BIO tagging with exact character offsets)
phase3/   rhetorical roles (hierarchical sentence classifier)
phase4/   judgment prediction (long-document classification, hierarchical)
phase5/   statute identification (multi-label, label-wise attention)
phase6/   summarization (chunk-level seq2seq)
phase7/   prior case retrieval (bi-encoders + BM25 hybrid)
```
Each `phaseN/` folder has a `README.md` (task, data findings, method, every experiment and its result, how to reproduce the best submission), the main notebooks, and `runs/` with the executed notebooks (outputs included) of every stage that was submitted.

## The Phase-1 foundation models (all implemented from scratch in PyTorch)

All: 12 layers of attention depth, 12 heads, d_model 768, context 1,024, vocabulary 32,768 (own BPE), tied embeddings, no dropout.

| Model | Design | Params |
|---|---|---|
| **A0** GPT-2 | learned positions, pre-LayerNorm, GELU MLP 3072, biases | 111.0M |
| **A1** modern decoder | RoPE, RMSNorm, SwiGLU (2048), no biases | 110.1M |
| **B** encoder–decoder | 6 bidirectional encoder + 6 causal decoder layers (cross-attention), random split point | 124.3M |
| **E** prefix-LM | A1 stack, bidirectional prefix of random length 256–768, causal continuation | 110.1M |
| **F** local–global | sliding window 256 in 9 layers + full attention every 4th layer, GQA (12 query / 4 KV heads), ALiBi | 107.8M |

Training: 1B tokens of a Supreme-Court-heavy mix (SC 80 %, Acts 10 %, High Courts 5 %, Tribunals 5 %), AdamW lr 6e-4, cosine schedule, 131k tokens/step, fp16, 2 × T4 (DDP), resumable across Kaggle sessions.

## How to run (all notebooks)

* Platform: **Kaggle** notebooks (also run on Colab / locally). Settings used: GPU **T4 × 2** for training notebooks, CPU for data notebooks; "Save & Commit" (batch) mode. All outputs go to `/kaggle/working/cs781_p1/` (Colab: `MyDrive/cs781_p1`).
* Data: downloaded inside the notebooks from the official Hugging Face dataset `Exploration-Lab/CS-781-Capstone` (add your HF token as the Kaggle secret `HF`, Internet ON for data notebooks).
* Every notebook starts with the same *shared setup cell* (environment detection, paths, artifact search across attached inputs, time budget). Every heavy step is checkpointed and resumes when the notebook is re-run with its previous output attached.
* Pipeline order: `phase1/01` → `phase1/02` → `phase1/03` (one run per architecture) → for each later phase `PN_01` (data, CPU) → `PN_02` (fine-tuning, GPU). Each `PN_02` takes as inputs the `PN_01` output and the five Phase-1 weight folders (`runs/<run_name>/weights_final.pt + run_info.json`).
* Experiment notebooks are **queues**: the configuration cell defines `EXPERIMENTS` (name → settings) and `QUEUES`; set `RUN_QUEUE = QUEUES["..."]` to choose what a commit runs (one worker thread per GPU, dependencies such as KD teachers respected). Each experiment writes `p<N>/exps/<name>/` with `config.json`, `result.json`, logs, trainable weights and `submission.zip`.
* Model weights are not included (size); they are produced by the notebooks.

## Techniques implemented (shared across phases)

Full fine-tuning · top-k layer fine-tuning · gradual unfreezing · frozen backbone (probe) · BitFit · LoRA · rsLoRA · LoRMA⁺ (I + α/r·BA)·W₀ and LoRMA-π · Houlsby adapters · Pfeiffer adapters · IA³ · prompt tuning · knowledge distillation (α·CE + (1−α)·T²·KL into a 6-layer student initialised from Phase-1 layers) · task heads: BiLSTM, CRF, constrained Viterbi, label-wise attention, hierarchical BiGRU, contrastive bi-encoder with gradient caching.

## References (main)
Vaswani et al. 2017; Radford et al. 2019 (GPT-2); Su et al. (RoPE); Zhang & Sennrich 2019 (RMSNorm); Shazeer 2020 (SwiGLU); Ainslie et al. 2023 (GQA); Press et al. 2022 (ALiBi); Beltagy et al. 2020 (Longformer); Raffel et al. 2020 (T5); Dong et al. 2019 (UniLM); Hu et al. 2022 (LoRA); Bihany et al. 2025 (LoRMA); Kalajdzievski 2023 (rsLoRA); Liu et al. 2022 (IA³); Houlsby et al. 2019; Pfeiffer et al. 2021; Lester et al. 2021; Ben Zaken et al. 2022; Howard & Ruder 2018; Hinton et al. 2015; Vu et al. 2020 (LAAT); Ridnik et al. 2021 (ASL); Yang et al. 2016 (HAN); Robertson & Zaragoza 2009 (BM25); Karpukhin et al. 2020 (DPR); Khattab & Zaharia 2020 (ColBERT); Gao et al. 2021 (GradCache); Hofstätter et al. 2021 (TAS-B); datasets: IL-TUR (Joshi et al. 2024), Kalamkar et al. 2022, Malik et al. 2021/2022, Paul et al. 2022, Shukla et al. 2022, Joshi et al. 2023 (U-CREAT / IL-PCR).
