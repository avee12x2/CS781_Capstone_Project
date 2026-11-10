# Phase 1: Legal language model pretraining

**Task.** Given a 512-word prompt taken from an Indian court judgment, generate the next 128 words. The official metric is BERTScore-F1 against the true continuation; ROUGE-L and BLEU-4 are reported as secondary metrics. Models are limited to about 110M parameters and a 1,024-token context. Training uses only the official corpus (Acts, High Courts, Supreme Court, Tribunals) and starts from scratch.

**Best test submission:** `A0_gpt2_110M_SC80_epoch_1000M`, BERTScore-F1 **0.8393**. The other four 1B-token models were close behind: F 0.8388, A1 0.8387, E 0.8382, B 0.8312.

## Notebooks

| Notebook | What it does | Kaggle settings | Inputs |
|---|---|---|---|
| `01_data_tokenizer_evalset.ipynb` | Reads the test prompts and makes training text look like them (whitespace collapsed, cased). Streams one statistics pass over each source: exact-duplicate removal (123,682 docs), a document-level train/val split, and a balanced 150 MB sample per source. Trains byte-level BPE at 16k / 32k / 49k and compares it with GPT-2 (fertility, prompt length, round trip). Includes an educational pure-Python BPE trainer. Builds the **local evaluation set** of 400 held-out Supreme-Court windows (512-word prompt + 128-word truth) and scores two trivial baselines. | CPU, Internet ON, secret `HF` | none |
| `02_encode_shards.ipynb` | Tokenizes the cleaned, deduplicated corpus once into `uint16` shards (`shards/bpe32768/<source>_<train|val>_<k>.bin`), with `<|endoftext|>` between documents. Resumable shard by shard. | CPU, Internet ON | NB1 output |
| `03_train_eval_submit.ipynb` | Contains all architectures (A0, A1, B, E, F, plus RetNet C and hybrid D, which were implemented but never trained), the mixed-source data loader with **random** or **epoch** sampling, DDP training on 2 GPUs, resumable checkpoints every 30 min, per-source validation loss, local BERTScore/ROUGE/BLEU, a decoding sweep, and `submission.zip`. One commit trains one architecture. | GPU T4 × 2, Internet ON | NB1 + NB2 outputs; its own previous output when resuming |
| `04_decode_lab.ipynb` | Decoding analysis: n-gram block, repetition-penalty grid, beam search, model vs copy-tail baseline, quality across snapshots. **Not executed.** The decoding results in the report come from the sweep inside NB3. | GPU T4 | NB1 output + one NB3 run |
| `history/` | Earlier NB3 versions (v1 A0 only; v2 A0+A1; v3 A0/A1/B) and the executed 300M and 1.1B-token A0 mix50 runs. | | |
| `run_logs/<run>/` | For each of the five final 1B-token runs: `run_info.json` (full config, parameter count), `summary.json` (final metrics, decoding sweep, timings) and `metrics.jsonl` (training/validation curves). | | |

## Architectures (`PRESET` in the NB3 config cell)

Every model has 12 layers of attention depth, 12 heads, d = 768, context 1,024, vocabulary 32,768, tied input/output embeddings and no dropout.

* **A0, GPT-2** (`arch="gpt2"`, 111.0M): learned positions, pre-LayerNorm, GELU MLP 3072, biases.
* **A1, modern** (`arch="modern"`, 110.1M): RoPE, RMSNorm, SwiGLU (2048), no biases. Each component can be switched back individually for ablations.
* **B, encoder–decoder** (`arch="encdec"`, 124.3M): 6 bidirectional encoder layers and 6 causal decoder layers with cross-attention. Each training window is split at a random point E ∈ [256, 768]: the encoder reads the first E tokens and the decoder predicts the rest. At evaluation the split is at 768.
* **E, prefix-LM** (`arch="prefixlm"`, 110.1M): the A1 stack, but the first E ∈ [256, 768] tokens attend bidirectionally. The loss is computed only on the causal continuation.
* **F, local–global** (`arch="efficient"`, 107.8M): sliding-window attention (256) in 9 layers and full attention in every 4th layer. Grouped-query attention with 12 query and 4 KV heads, ALiBi positions, FFN 2304. Its KV cache is 3× smaller than A1's.

## Training recipe of the final runs

* **Data mix "SC80":** `mix = {supreme_court: 0.8, acts: 0.1, high_courts: 0.05, tribunals: 0.05}`. The test prompts are much closer to Supreme-Court text (JSD 0.032) than to any other source (≥ 0.151).
  * A0 and A1 were run with `supreme_court: 0.9`, which normalizes to 81.8 / 9.1 / 4.5 / 4.5 %.
* **Sampling:** `sampling="epoch"`. Each source is cut into windows that are shuffled and used once before any window repeats. 1B tokens means 3.2 passes over the 253M SC tokens.
* **Optimization:**
  * `token_budget = 1_000_000_000`, which is 7,629 steps of 131,072 tokens (2 GPUs × micro-batch 8 × 8 accumulation steps);
  * AdamW with lr 6e-4, betas (0.9, 0.95) and weight decay 0.1;
  * 2 % warm-up, then cosine decay to 10 %;
  * gradient clipping 1.0;
  * fp16 with dynamic loss scaling.
* **Decoding:** left-truncate the prompt, ban special tokens, greedy decoding with no-repeat-4-gram and repetition penalty 1.15, then cut the output to exactly 128 words.
  * Greedy decoding alone loops: 56–64 % of its 4-grams are repeats. The 4-gram block adds ≈ +0.02 BERTScore and the penalty another +0.003.
* **Wall-clock time on 2 × T4:** A0 8.4 h, A1 11.4 h, B 6.8 h, E 11.3 h, F 12.2 h. Training is resumable across 12-hour sessions.

## Results

| Run | Val loss (SC) | Local BERTScore | Test BERTScore |
|---|---|---|---|
| A0 GPT-2 SC80 1B | 2.277 | 0.842 | **0.8393** |
| A1 modern SC80 1B | 2.282 | 0.842 | 0.8387 |
| B enc–dec SC80 1B | 2.519* | 0.834 | 0.8312 |
| E prefix-LM SC80 1B | 2.344* | 0.842 | 0.8382 |
| F local–global SC80 1B | **2.251** | 0.843 | 0.8388 |
| A0 mix50 300M (first submission) | – | – | 0.8351 |

\* Measured only on the continuation after a 768-token prefix, so not comparable with the other rows.

**Findings:**

* More tokens was the biggest lever: 300M → 1B tokens gave +0.004.
* SC-heavier mixes and epoch sampling each gave smaller gains.
* Copying the last 128 prompt words scores 0.842 locally, slightly above every model. BERTScore rewards reusing the prompt's case-specific vocabulary.

## How to reproduce the best submission
1. Run `01_data_tokenizer_evalset.ipynb` (CPU). This produces `tokenizer/bpe32768`, `data/` and `eval/`.
2. Run `02_encode_shards.ipynb` (CPU) with the NB1 output attached. If it runs out of time, re-run it with its own previous output attached.
3. In `03_train_eval_submit.ipynb`, set:
   ```python
   PRESET = "A0"
   PRESETS["A0"]["run_name"] = "A0_gpt2_110M_SC80_epoch_1000M"
   CFG.update(mix={"supreme_court": 0.9, "tribunals": 0.05, "high_courts": 0.05, "acts": 0.1},
              sampling="epoch", token_budget=1_000_000_000)
   ```
   Commit on GPU T4 × 2 with NB1 and NB2 attached. When a commit hits the time limit, re-commit with the previous output attached and training resumes from the last checkpoint.
4. At the end, the notebook evaluates on the 400 local prompts, sweeps decoding and writes `runs/<run_name>/submission.zip` using greedy + nr4 + rp1.15.
5. `runs/<run_name>/weights_final.pt` and `run_info.json` are the Phase-1 checkpoint that every later phase loads.

To train the other models, set `PRESET` to `"A1"`, `"B"`, `"E"` or `"F"`, use the same `CFG` overrides with SC weight 0.8, and set the matching `run_name` from `run_logs/`.
