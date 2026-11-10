# Phase 3: Rhetorical role prediction

**Task.** Assign one of 13 rhetorical roles to every sentence of a judgment. Examples: Fact, ArgumentPetitioner, ArgumentRespondent, RatioOfTheDecision, PrecedentReliedUpon, RulingByLowerCourt, RulingByPresentCourt, Dissent, PrecedentOverruled. The data covers two domains, Competition Law (CL) and Income Tax (IT). Metric: macro-F1 over the 13 roles.

**Best test submission:** `f3_F_seppool_ia3`. F backbone with causal attention; each sentence is represented by the state at its separator token; IA³ trains ≈ 2 % of the parameters; retrained on train + dev. Macro-F1 **0.5749**.

## Data findings (P3_01)
* 100 judgments. Each domain has 40 / 5 / 5 documents for train / dev / test, giving 17k training sentences.
* Roles are sequential: a sentence has the same role as the previous one 87 % of the time.
* Roles are also positional: RulingByPresentCourt lies in the last 11 % of the document.
* Dissent and PrecedentOverruled almost never occur and are absent from dev, which caps macro-F1 at 11/13 = 0.846.
* The provided sentence lists are kept and packed whole into 512-token windows (1.9 % of sentences are truncated at 128 tokens).

## Method (P3_02)
1. **Windowing.** Chunks of 48 consecutive sentences are packed into 512-token windows, so each sentence sees its neighbours' tokens.
2. **Sentence vector.** Either the mean of the sentence's token states (bidirectional backbones) or the state at the sentence's end separator (`seppool`). For causal backbones the separator has read the whole sentence.
3. **Extra embeddings.** A relative-position embedding (32 buckets) and a CL/IT domain embedding are added.
4. **Sequence model.** A BiLSTM runs over the sentences of the chunk, followed by a linear or CRF head.
5. **Inference.** Chunks overlap by half and predictions are averaged with centre weights.
6. **Training.** CL and IT are trained jointly (per-domain models were no better), with square-root inverse class weights and dev-tuned per-label biases. Final models are retrained on train + dev.

## Experiments (`QUEUES` in `P3_02_rr_finetune.ipynb`)
| Queue | Experiments | Question / finding |
|---|---|---|
| `stageA` | `a1_E_sentence_only`, `a2_E_window`, `a3_E_window_lstm_pos`, `a4_E_window_lstm_crf`, `a5_E_CL_only`, `a6_E_IT_only`, `a7_A1_random_init` | Context ladder on test: sentence alone 0.380 → window 0.513 → + BiLSTM + position 0.564. CRF did not help, nor did per-domain models. Random init → Phase-1 init: 0.297 → 0.517. |
| `stageB` | `b1_A0_causal`, `b2_A1_causal`, `b3_F_causal`, `b4_B_encoder`, `b5_A1_causal_seppool` | Backbone comparison, and separator pooling for causal backbones. |
| `stageC` | `c1_E_frozen`, `c2_E_lora`, `c3_E_rslora`, `c4_E_lorma`, `c5_E_lorma_pi`, `c6_E_ia3`, `c7_E_prompt` | Adaptation techniques (frozen → full FT on test: 0.468 → 0.564). |
| `final_acc1` | `f4_B_kd`, `f2_A1_seppool_lora`, `f6_A1_seppool_ctx96` | 6-layer B-encoder KD student (teacher from stage B), A1 + LoRA, and A1 with 96-sentence context. |
| `final_acc2` | `f1_E_lorma`, `f3_F_seppool_ia3`, `f5_A0_seppool_rslora` | Final E + LoRMA⁺, F + IA³ (**best**), and A0 + rsLoRA. |

## Test results (23 submissions; main variants)
| Submission | Macro-F1 |
|---|---|
| F + IA³, separator pooling (`f3`) | **0.5749** |
| E full FT with context (stage A) | 0.5636 |
| A0 + rsLoRA (`f5`) | 0.5587 |
| E + LoRMA⁺ (`f1`) | 0.5173 |
| B 6-layer KD student (`f4_B_kd`) | 0.5094 (0.464 without KD) |

**Common confusions:**

* ArgumentPetitioner ↔ ArgumentRespondent
* RatioOfTheDecision ↔ PrecedentReliedUpon
* Fact ↔ RulingByLowerCourt

CL is consistently harder than IT.

## How to reproduce the best submission
1. Run `P3_01_rr_data.ipynb` (CPU, Internet ON, secret `HF`).
2. Run `P3_02_rr_finetune.ipynb` on GPU T4 × 2 with these inputs: the P3_01 output and the five Phase-1 run folders.
3. Set `RUN_QUEUE = QUEUES["final_acc2"]`. `f3_F_seppool_ia3` writes `p3/exps/f3_F_seppool_ia3/submission.zip`.

`runs/` holds the executed notebooks of stages A–C and both final accounts.
