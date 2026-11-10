# Phase 6: Abstractive summarization (IN-Abs)

**Task.** Write an abstractive summary of a full Indian Supreme Court judgment. The primary metric is BERTScore-F1; ROUGE-1/2/L are secondary.

**Best test submission:** `A1_lora`, BERTScore-F1 **0.8509**.
* **Model:** the A1 backbone with LoRA (r 16, α 32, all linear layers, 2.35 % trainable), trained on 20k chunk–summary pairs for 2 epochs.
* **Generation:** chunk-level, greedy with no-repeat-4-gram and repetition penalty 1.1.
* `F_houlsby_40k` ties at 0.85091.

## Data findings (P6_01)
* 7,030 judgment–summary pairs. We held out 355 by an ID hash for validation (there is no official dev set). The test set has 100 documents.
* Median length is 3,131 words per judgment and 639 per summary, a compression of 0.22. Fitted length model: `summary_words ≈ 4.62 · doc_words^0.612`.
* Summary sentences align well to source chunks (median alignment score 0.83), and 22 % come from the first tenth of the judgment.
* **Alignment.** P6_01 splits each judgment into sentence-aligned chunks of ≤ 600 tokens. Each summary sentence is assigned to the chunk with the highest overlap score: 0.7 · IDF-weighted content-word recall + 0.3 · bigram recall. This gives 47.5k non-empty pairs (chunk → chunk summary ≤ 384 tokens). The notebook also computes extractive baselines.

## Method (P6_02)
* **Decoder-only models (A0, A1, F)** are trained on `chunk Summary: summary<eos>`, with loss on the summary tokens only.
* **E** attends bidirectionally over the chunk and causally over the summary.
* **B** encodes the chunk and decodes the summary.
* **Test time:**
  1. Each chunk gets a word budget proportional to its share of the expected summary length (from the power law).
  2. Chunk summaries are generated greedily with a KV cache, no-repeat-4-gram and repetition penalty 1.1.
  3. They are joined in document order and cut at `len_scale ×` the expected length.
* **Techniques:**
  * LoRA, LoRMA⁺;
  * Houlsby adapters (on all residual outputs), Pfeiffer adapters (on FFN outputs);
  * full fine-tuning as a reference;
  * token-level KD into a 6-layer A0, with a no-KD control.
* No post-training or alignment method (RLHF, DPO, etc.) was used.

## Experiments (`QUEUES` in `P6_02_summ_finetune.ipynb`)
| Queue | Experiments | Status / finding |
|---|---|---|
| `acc1` | `A1_lora`, `E_lorma`, `A0_6L_kd`, `E_lorma_len0.6`, `E_lorma_len1.6`, `E_lorma_norep` | Run. Full FT vs LoRMA⁺ on E (val): 0.8545 vs 0.8541. Without repetition control, 36 % of trigrams repeat. Longer summaries raise ROUGE but not BERTScore, which only reads ≈ 380 words. |
| `acc2` | `F_houlsby`, `E_full`, `B_pfeiffer`, `A0_6L_control` | Run. B with FFN-only adapters underfits. KD did not improve the generated text (val 0.8507 vs 0.8497). |
| `acc2_r2` | `F_houlsby_40k`, `F_houlsby_front`, `F_houlsby_len1.6`, `E_full_front`, `A0_6L_kd_Efull` | Run. 20k → 40k training pairs gave +0.015 ROUGE-1. |
| `acc1_r3`, `acc2_r3` | `*_front`, `*_len1.6` variants | Defined but not run (deadline). |

## Test results (4 submissions)
| Submission | BERTScore-F1 | ROUGE-1 |
|---|---|---|
| A1 + LoRA (`A1_lora`) | **0.8509** | 0.565 |
| F + Houlsby adapters, 40k pairs (`F_houlsby_40k`) | 0.8509 | 0.570 |
| E + LoRMA⁺, longer summaries | 0.8477 | **0.586** |
| 6-layer A0 KD student (`A0_6L_kd`) | 0.8465 | 0.559 |

Generated summaries are fluent and on topic but shorter than the references: 511–673 words against 746.

## How to reproduce the best submission
1. Run `P6_01_summ_data.ipynb` (CPU, Internet ON, secret `HF`). It produces chunk-aligned pairs, the validation split and the length model.
2. Run `P6_02_summ_finetune.ipynb` (GPU T4 × 2) with the P6_01 output and the Phase-1 run folders attached, and `RUN_QUEUE = QUEUES["acc1"]`. The checked-in notebook is set to `acc1_r3`; change it to `acc1`.
3. `A1_lora` trains, generates validation and test summaries, and writes `p6/exps/A1_lora/submission.zip`. `A0_6L_kd` waits for its teacher `E_lorma` from the same queue.

`runs/` contains the executed `acc1`, `acc2` and `acc2_r2` notebooks.
