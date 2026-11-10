# Phase 4: Court judgment prediction (CJPE / ILDC)

**Task.** Read a Supreme Court judgment with its decision removed and predict whether the appeal was accepted (1) or rejected (0). The primary metric is macro-F1; precision and recall of class 1 are secondary.

**Best test submission:** `s6_hier16_E_chunks4`, macro-F1 **0.796** (P 0.811 / R 0.776). It is built in two steps:
1. Fine-tune the E (prefix-LM) encoder end-to-end on the last 4 × 512-token chunks.
2. Freeze it and embed up to 16 chunks (8,192 tokens), then train a hierarchical BiGRU with attention over the chunk vectors. The threshold is tuned on clean dev: 0.41, dev 0.8168.

## Data findings (P4_01)
* **Duplicates.** Repeated IDs are exact copies, so we dropped 5,082 in train and 994 in dev. We also removed 653 exact-text duplicates, which leaves **31,652 training documents**.
* **Dev/test overlap.** The 3,505-record dev file contains all 1,517 test documents with labels. We used only the **994 dev documents that are not test documents** for model selection and thresholds, and never trained on dev.
* **Class balance.** Train is 41 % accepted; dev and test are balanced. Thresholds are therefore tuned on dev.
* **Casing.** Dev and test are 99 % lower-case, so all splits are lower-cased.
* **Length.** 40 % of dev/test documents exceed 4,096 tokens.
* **Where the signal is.** A TF-IDF probe finds it at the end of the document: the last 512 tokens give 0.655 macro-F1, the first 512 give 0.576. P4_01 also saves the tokenized documents as `uint16` memmaps.

## Method (P4_02)
* **Representations:**
  * head, middle or tail 512 tokens, with mean, last-token or attention pooling;
  * tail 1,024 tokens;
  * end-to-end training on the last 4 × 512 chunks with attention pooling over the chunk vectors;
  * hierarchical: a frozen fine-tuned encoder embeds 8 or 16 chunks by mean pooling, then a BiGRU with attention runs over them.
* **Training:** class-balanced cross-entropy, with the decision threshold tuned on clean dev.
* **Techniques:**
  * full fine-tuning, top-4 layers, gradual unfreezing, frozen backbone, BitFit, Houlsby adapters;
  * LoRMA⁺ and LoRMA-π;
  * KD into a 6-layer A1 student (α = 0.5, T = 2), with a no-KD control.

## Experiments (`QUEUES["stage1".."stage7"]` in `P4_02_cjpe_finetune.ipynb`)
| Stage | Experiments | Finding |
|---|---|---|
| 1 | `s1_E_{head512,middle512,tail512,headtail512,tail1024,tail512_attnpool,chunks4_attn}`, `s1_A1_random_init` | The tail is the best single window. Random → Phase-1 init: test 0.593 → 0.716. |
| 2 | `s2_{A0,A1,F}_causal`, `s2_B_encoder`, `s2_E` | Backbones on tail 512. |
| 3 | `s3_E_{frozen,bitfit,top4,gradual,adapter,lorma,lorma_pi}` | Frozen → full: 0.588 → 0.703. Full vs LoRMA⁺: 0.703 vs 0.707. LoRMA-π on attention only: 0.608. |
| 4 | `s4_A1_6L_{kd,control}`, `s4_B_kd`, `s4_{A0,F}_lorma`, `s4_E_chunks4_all`, `s4_hier8_{A0,A1,E,F}`, `s4_hier16_E` | KD: 0.708 → 0.715. 4-chunk end-to-end E reaches 0.786. Hierarchy over 8 and 16 chunks. |
| 5 | `s5_hier16_{A0,A1,A1_6L_kd,E_lorma,F_lorma}`, `s5_A0_lorma_attn` | 16-chunk hierarchy on every backbone. Backbones end up within ≈ 0.02 of each other. |
| 6 | `s6_hier16_E_{chunks4,gradual,top4}` | Hierarchy on top of the 4-chunk encoder (**best, 0.796**). |
| 7 | `s7_{A0,A1,F}_top4`, `s7_hier16_{A0,A1,F}_top4`, `s7_hier16_F` | Partial fine-tuning on the other backbones. |

## Test results (main variants)
| Submission | Phase-1 model / technique | Macro-F1 | P / R |
|---|---|---|---|
| `s6_hier16_E_chunks4` | E, full FT → hierarchy | **0.796** | 0.811 / 0.776 |
| `s5_hier16_A0` | A0, full FT → hierarchy | 0.784 | 0.801 / 0.760 |
| `s5_hier16_F_lorma` | F, LoRMA⁺ → hierarchy | 0.761 | 0.768 / 0.751 |
| `s5_hier16_A1_6L_kd` | 6-layer A1, KD from E → hierarchy | 0.756 | 0.741 / 0.794 |

The published ILDC hierarchical XLNet + BiGRU scores ≈ 0.78. Most errors are in long judgments, and false negatives are made with high confidence.

## How to reproduce the best submission
1. Run `P4_01_cjpe_data.ipynb` (CPU, Internet ON, secret `HF`).
2. Run `P4_02_cjpe_finetune.ipynb` (GPU T4 × 2) with the P4_01 output and the five Phase-1 run folders attached, and `RUN_QUEUE = QUEUES["stage4"]`. Only `s4_E_chunks4_all` is needed for the best model; it trains the 4-chunk E encoder.
3. Re-run P4_02 with the stage-4 output also attached and `RUN_QUEUE = QUEUES["stage7"]`. This queue starts with `s6_hier16_E_chunks4`, which embeds 16 chunks with the frozen encoder, trains the BiGRU and writes `p4/exps/s6_hier16_E_chunks4/submission.zip`.

`runs/` contains the executed notebooks of all seven stages.
