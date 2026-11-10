# Phase 2: Legal Named Entity Recognition (L-NER)

**Task.** Find the entities in Indian judgments (IL-TUR L-NER, 12 types: APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO) and return exact character offsets. The metric is **strict** entity-level macro-F1: a span that is one character off counts as wrong.

**Best test submission:** `f1_A1_causal_bilstm_full`. A1 backbone, causal attention as in pretraining, BiLSTM head, full fine-tuning, retrained on all 70 training documents. Strict macro-F1 **0.6155**. The published IL-TUR baseline is 0.486.

## Data findings (P2_01)
* 105 judgments in three folds of 35. Folds 1–2 (70 documents, 8,393 entities) are given; fold 3 is the test set. We held out 14 documents for development.
* Documents average 6.7k tokens and 98 % exceed 1,024.
* Entity types are imbalanced: DATE has 1,575 mentions and R.COUNSEL 184; witness labels occur in only 12 documents.
* Party names cluster in the first 5 % of a judgment.
* There are no nested spans, and a string never receives two labels within one document. Only 79–99 % of repeated mentions are annotated.
* **Offsets:**
  * Each document is whitespace-normalized with a character index map, and punctuation is split from words before BPE, which lifts the oracle tagger ceiling from 0.986 to 0.994.
  * Text is tagged in BIO and decoded back through both maps.
  * Only 1 of 8,393 gold spans fails the round trip, and a unit test checks that gold tags reproduce gold spans.

## Method (P2_02)
* **Model:** a Phase-1 backbone (LM head removed, pretraining attention kept) followed by a head: linear, BiLSTM over the backbone states, or CRF. BIO tagging uses 25 classes.
* **Windows:** 512-token windows, each with the document header (first 256 tokens: parties, judges, counsel) prepended. At inference, windows overlap by half and each token keeps the prediction from its most central window.
* **Class imbalance:** square-root inverse class weights capped at 10, plus oversampling of windows that contain rare labels.
* **Decoding:**
  * Viterbi search constrained to valid BIO transitions.
  * Per-label decision biases.
  * Document-level label consistency: a string keeps its majority label within the document.
  * Each step is kept only if it improves dev F1 by ≥ 0.005.
* **Final models:** retrained on all 70 documents for the dev-selected number of epochs.

## Experiments (`EXPERIMENTS` / `QUEUES` in `P2_02_finetune_ner.ipynb`)
| Queue | Experiments | Question |
|---|---|---|
| `stage1` | `s1_{A0,A1,F}_full_{causal,bidir}`, `s1_B_full`, `s1_E_full`, `s1_A1_randominit` | Which backbone works best, should attention be causal or bidirectional, and how much does pretraining help? Random init → Phase-1: dev .365 → .524. Removing the causal mask hurt A0 (−0.11) and F (−0.08). |
| `stage2` | `s2_E_{bilstm,crf,crf_header128,header128,header256,noclassw,oversample,win1024}`, `s2_A1_causal_bilstm` | Heads, header context, class weights, oversampling, window size. The +256-token header gave dev .556 → .580. A BiLSTM gave causal models right-hand context (.540 → .589). A CRF raised precision but not F1. |
| `stage3` | `s3_E_{frozen,bitfit,top4,adapter64,lora16,lorma16,lorma_pi}` | Adaptation techniques on E. Frozen → full: .414 → .556. LoRMA⁺ matched full fine-tuning. |
| `final_acc1` | `f1_A1_causal_bilstm_full`, `f4_B_kd`, `f4_B_nokd` | Final retrained A1 model, and a 6-layer B-encoder student with KD from the A1 + BiLSTM teacher (control without KD: dev .387 → .416). |
| `final_acc2` | `f2_E_lorma`, `f3_F_causal_adapter` | Final E + LoRMA⁺ and F + Houlsby adapters + BiLSTM. |

## Test results (7 submissions)
| Submission | Strict macro-F1 |
|---|---|
| A1 + BiLSTM, full FT (`f1`) | **0.6155** |
| F adapters + BiLSTM (`f3`) | 0.6068 |
| E LoRMA⁺ (`f2`) | 0.6013 |
| E full FT (stage 1/2) | 0.5967 |
| B 6-layer KD student (`f4_B_kd`) | 0.4641 |

The hard labels are APP, RESP, WIT and AUTH. Most errors are PREC boundary mistakes on long citations and unannotated repeated names.

## How to reproduce the best submission
1. Run `P2_01_data_alignment.ipynb` (CPU, Internet ON, secret `HF`).
2. Run `P2_02_finetune_ner.ipynb` on GPU T4 × 2 with these inputs:
   * the P2_01 output;
   * the five Phase-1 run folders (`runs/<run>/weights_final.pt`, `run_info.json`);
   * for `f4_B_kd` only, the Stage-2 output, which holds the KD teacher.
3. Set `RUN_QUEUE = QUEUES["final_acc1"]`. `f1_A1_causal_bilstm_full` writes `p2/exps/f1_A1_causal_bilstm_full/submission.zip`.

`runs/` contains the executed notebooks of stages 1–3 and the two final accounts.
