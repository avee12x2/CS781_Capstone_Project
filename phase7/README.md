# Phase 7: Prior case retrieval (IL-PCR)

**Task.** For each query judgment, whose own citations are masked as `<CITATION>`, rank the candidate pool and return the 7 cases most likely to be cited. The primary metric is Micro-F1@7; P@7 and R@7 are secondary.

**Best test submission:** `E_lorma_r2` (hybrid), Micro-F1@7 **0.4761** (P@7 0.486 / R@7 0.467, dev 0.4836). It is the E backbone trained as a LoRMA⁺ bi-encoder (r 16, 2.69 % trainable) for 8 epochs with 3 BM25 hard negatives per query. Its scores are fused with our lexical++ BM25 and a time rule. `A1_full_ctx` ties at 0.4761.

## Data findings (P7_01)
* Train / dev / test have 827 / 118 / 237 queries, with pools of 4,320 / 1,023 / 1,727 cases. Every query is inside its own pool, so it is excluded from its own ranking.
* Queries cite 6.4 (train) and 8.3 (dev) pool cases on average, up to 92. A perfect top-7 system therefore reaches only 0.618 Micro-F1@7 on dev.
* 82 % of cited cases are cited only once, so a popularity prior is useless.
* BM25 puts only 38 % of cited cases in its top 7 and 16 % beyond rank 200.
* Supreme Court headers carry `DATE OF JUDGMENT` and `CITATOR INFO`. The latter lists *later* cases that cite this one, so it is ignored.
* Test relevance labels are never read.

## Method
**Dense bi-encoder (P7_02).**
* **Encoding.** Each Phase-1 model becomes a shared-weight encoder. A case is split into ≤ 512-token sentence-aligned chunks, plus one window around each `<CITATION>`. Query and candidate use the same encoder and policy, with no special tokens. A chunk vector is the L2-normalized mean of its token states.
* **Scoring** has three variants:
  * cosine of mean chunk vectors;
  * MaxSim, where each query chunk takes its best candidate chunk and the scores are averaged;
  * citation-window scoring (`qview="ctx"`): the best match between any query citation window and the candidate.
* **Training:**
  * InfoNCE with τ = 0.05 over in-batch negatives plus 1–3 BM25 hard negatives, with the query's other cited cases masked out of the denominator;
  * three random chunks per case per step;
  * gradient caching, which fits batches of ≈ 120 long chunks on a T4;
  * for citation windows, a multiple-instance loss: a cited case only needs to match its best window.
* **Techniques:**
  * zero-shot (`zs_*`), full FT, top-k on B, LoRMA⁺, IA³;
  * KD into a 6-layer A0 student, which learns the teacher's softmax over each batch's candidates from a precomputed teacher score matrix;
  * a no-KD control.

**Lexical++ and hybrid.**
* **Lexical++.** Our own sparse BM25: a fixed formula whose statistics come from the split's own pool, with no learned weights. It runs over 1-, 2- and 3-grams of the full query and of the citation windows, with raw or binary query weights. These 12 features are weighted by coordinate ascent on train F1@7.
* **Hybrid ranking:** `z(dense) + λ · z(lexical) − γ · 1[candidate year > query year]`, with per-query z-scores. λ, γ and the scoring variant are chosen on dev; with λ = 0 the score is simply `z(dense)`.
* **Case year.** Taken from `DATE OF JUDGMENT`, or otherwise the latest year mentioned outside `CITATOR INFO`.

## Experiments (`QUEUES` in `P7_02_pcr_finetune.ipynb`)
| Round / queue | Experiments | Finding |
|---|---|---|
| 1 `acc1` | `E_lorma`, `zs_{E,B,A1,A1_bidir,F,A0}`, `B_topk`, `A0_6L_kd` | Zero-shot dense retrieval is weak. Full-query bidirectional attention for causal models hurt (A1 0.305 → 0.268). |
| 1 `acc2` | `A1_full`, `F_ia3`, `A0_6L_control`, `E_full` | Frozen → full (dense only, dev): 0.279 → 0.355. Full vs LoRMA⁺ on E: 0.367 vs 0.368. KD: 0.359 → 0.371. |
| 2 `acc1_r2`, `acc2_r2` | `*_r2`: 8 epochs, 3 hard negatives | Superseded by round 3. `E_lorma_r2` was run inside `acc1_r3`. |
| 3 `acc1_r3` | `E_lorma_ctx`, `E_lorma_rs`, `A0_6L_kd_rs`, `E_lorma_r2`, `A0_6L_kd_ctx` | Citation windows: dense dev 0.368 → 0.390. Lexical++ with the time rule: 0.416 → 0.476. Dense models now add only +0.005–0.01 on top. |
| 3 `acc2_r3` | `A1_full_ctx`, `F_ia3_ctx`, `A1_full_rs`, `F_ia3_rs`, `A0_6L_control_ctx` | Same comparisons on A1 and F. |

`extra_P7_03_pcr_fuse_not_run.ipynb` (per-citation BM25 and re-fusion) was written but not run before the deadline.

**Failed approaches:**
* Whole-document dense retrieval alone stayed below BM25 (dev 0.37 vs 0.42).
* A citation-popularity prior did not help.

## Test results (8 submissions, all hybrid)
| Submission | Model / technique | Micro-F1@7 |
|---|---|---|
| `E_lorma_r2` (round 3) | E, LoRMA⁺ | **0.4761** |
| `A1_full_ctx` (round 3) | A1, full FT | 0.4761 |
| `A0_6L_kd_ctx` (round 3) | 6-layer A0, KD from `E_lorma_ctx` | 0.4749 |
| `F_ia3_ctx` (round 3) | F, IA³ | 0.4725 |
| `E_lorma` (round 1) | E, LoRMA⁺ | 0.4241 |
| `A1_full` (round 1) | A1, full FT | 0.4206 |
| `A0_6L_kd` (round 1) | 6-layer A0, KD | 0.4188 |
| `F_ia3` (round 1) | F, IA³ | 0.4105 |

## How to reproduce the best submission
1. Run `P7_01_pcr_data.ipynb` (CPU, Internet ON, secret `HF`). It parses the data, runs the checks, extracts years, computes BM25 / lexical++ scores and BM25 hard negatives, and tokenizes chunks and citation windows. `runs/P7_01_v1_run.ipynb` is the executed first version; the main file is v2, which adds lexical++.
2. Run `P7_02_pcr_finetune.ipynb` (GPU T4 × 2) with the P7_01 output and the Phase-1 run folders attached, and `RUN_QUEUE = QUEUES["acc1_r3"]`.
   * `E_lorma_r2` has no dependencies.
   * The KD students need their teacher (`E_lorma_ctx` from the same queue, or `E_lorma` from round 1).
3. The experiment selects λ, γ and the scoring variant on dev and writes `p7/exps/E_lorma_r2/submission_hybrid.zip`.

`runs/` contains the executed round-1 and round-3 notebooks for both accounts.
