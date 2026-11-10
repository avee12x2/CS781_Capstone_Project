# Phase 5: Legal statute identification (ILSI)

**Task.** Given the facts section of a case, predict which of 100 statutes apply (multi-label). The primary metric is macro-F1; micro-F1 and subset accuracy are secondary.

**Best test submission:** `s4_E_chunks8_lwan_asl` with per-statute thresholds, macro-F1 **0.4107** (micro 0.505). The model is the E backbone, fully fine-tuned on 8 × 512-token chunks spread over the document. It uses a label-wise attention head and asymmetric loss.

## Data findings (P5_01)
* Splits are 42,750 / 10,181 / 13,019 facts sections. Merging 435 duplicates leaves 42,315 training documents. 108 train and 96 dev documents identical to test texts were removed from training and tuning.
* Documents carry 3.7 labels on average, and statute frequencies span a 43× range.
* Labels co-occur strongly, e.g. P(13 | 12) = 0.81.
* Section numbers are masked, so there is no lexical shortcut.
* Over half of the documents exceed 512 tokens.

## Method (P5_02)
* **Head (LAAT-style label-wise attention).** Every statute has its own query vector that attends over all token states of all chunks. Each statute then gets an independent sigmoid. A pooled linear head is the baseline.
* **Losses:** BCE, positive-weighted BCE, focal, and ASL (γ⁻ = 4, clip 0.05).
* **Input:** first 512 tokens, last 512 tokens, or 4 / 8 chunks of 512 spread evenly over the document. We also tried a hierarchical variant: 16 chunks → BiGRU → label attention.
* **Decision rules:** 0.5, one tuned global threshold, a tuned threshold per statute, top-k, and co-occurrence stacking. They are compared with **2-fold cross-fitting on dev**, so the dev score is never tuned on itself. Submissions use per-statute thresholds fitted on all of dev.
* **Techniques:** full FT, top-4, gradual unfreezing, frozen, BitFit, adapters, prompt tuning, LoRMA⁺. KD into a 6-layer A1 student is compared against a no-KD control.

## Experiments (`QUEUES` in `P5_02_lsi_finetune.ipynb`)
| Queue | Experiments | Finding |
|---|---|---|
| `stage1` | `s1_E_head512_{linear_bce,linear_bcepos,linear_focal,linear_asl,lwan_asl}`, `s1_E_tail512_lwan_asl`, `s1_E_chunks4_{attn,lwan}_asl` | After per-statute thresholds the losses are within 0.005. At threshold 0.5, BCE and focal collapse to 0.25 while ASL keeps 0.36. Label attention beats the pooled head by +0.006. |
| `stage2` | `s2_{A0,A1,B,F}` | Backbones. |
| `stage3` | `s3_E_{frozen,bitfit,top4,gradual,adapter,prompt,lorma}`, `s3_A1_random_init` | Random → Phase-1 (dev): 0.283 → 0.378. Frozen → full: 0.318 → 0.379. Full vs LoRMA⁺: 0.379 vs 0.365. |
| `stage4a` | `s4_A1_6L_ch4_{kd,control}`, `s4_F_ch4_lorma`, `s4_A0_ch4` | KD on test: 0.395 → 0.399. |
| `stage4b` | `s4_hier16_{A0,A1,E,F,E_lorma,E_chunks4}` | The hierarchy never beat end-to-end chunks (dev 0.396): mean-pooled chunk vectors blur the few phrases that signal a statute. |
| `stage4c` | `s4_E_chunks4_lwan_asl_5ep`, `s4_E_chunks8_lwan_asl` | More context helps: 512 → 4 chunks → 8 chunks gives 0.374 → 0.402 → **0.411**. |
| `stage5` | `s5_A0_ch4_{gradual,adapter}` | Partial adaptation on A0. |

## Test results (9 submissions; main variants)
| Submission | Macro-F1 |
|---|---|
| E, 8 chunks, LAAT + ASL (`s4_E_chunks8_lwan_asl`) | **0.4107** |
| A0, 4 chunks (`s4_A0_ch4`) | 0.405 |
| E, 4 chunks | 0.402 |
| 6-layer A1 KD student (`s4_A1_6L_ch4_kd`) | 0.399 |
| F + LoRMA⁺ (`s4_F_ch4_lorma`) | 0.399 |
| A0 gradual unfreezing (`s5_A0_ch4_gradual`) | 0.395 |

The published LeSICiN score is 0.285.

* **Per-statute vs global threshold:** per-statute thresholds beat one global threshold by +0.010 macro-F1 but lose 0.012 micro-F1.
* **Errors:**
  * Hub statutes (97, 43, 5) cause most false positives.
  * Statute 39 is missed most often.
  * F1 correlates with training frequency (Spearman 0.55).

## How to reproduce the best submission
1. Run `P5_01_lsi_data.ipynb` (CPU, Internet ON, secret `HF`).
2. Run `P5_02_lsi_finetune.ipynb` (GPU T4 × 2) with the P5_01 output and the five Phase-1 run folders attached, and `RUN_QUEUE = QUEUES["stage4c"]`. This is exactly the setting of `runs/P5_02_stage4c_run.ipynb`.
3. `s4_E_chunks8_lwan_asl` trains for 3 epochs, fits per-statute thresholds on dev and writes `p5/exps/s4_E_chunks8_lwan_asl/submission.zip`.

`runs/` contains the executed notebooks of stages 1, 2, 3, 4a, 4b, 4c and 5. The main notebook is the final (stage 5) version, which can run every queue.
