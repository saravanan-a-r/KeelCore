# KeelCore

**A compact base model for LoRA adapters, built to run in your web browser.**

A keel is the spine of a ship, the part everything else is fitted to. KeelCore
plays that role for text models. It is a 752-million-parameter encoder–decoder
Transformer, pretrained once from scratch on English prose, source code and
command-line text. Each capability is then added on top as a small **LoRA
adapter**, so a new task is a small download rather than a new model.

> ### EchoMeBetter is not the only use case of KeelCore. It is a by-product.
>
> **KeelCore is a general base model for LoRA adapters.** EchoMeBetter, the
> browser extension, is the first product built on it: it runs KeelCore's
> text-rewriting adapters on your device. That is one application of the base,
> not its purpose. More adapters, for different use cases, are in progress,
> and each one is added to the same base.

---

## One base, many adapters

```
                     ┌───────────────────────────────┐
                     │   KeelCore base  ·  752M      │   downloaded once
                     └───────────────┬───────────────┘
              ┌──────────────────────┼──────────────────────┐
      ┌───────▼───────┐      ┌───────▼───────┐      ┌───────▼───────┐
      │   adapter A   │      │   adapter B   │      │   adapter C   │   chosen by
      └───────────────┘      └───────────────┘      └───────────────┘   the user
```

- **Install the base once, then pick the adapters you need.** EchoMeBetter
  reads a catalogue (`model.json`) that lists the base and the adapters on
  offer, with each file's size and SHA-256 checksum. The base comes first;
  any set of adapters can then be added to it or removed again, each
  independently of the others.
- **Adapters are small.** A rank-16 LoRA adapter on every linear layer trains
  18,287,616 parameters, 2.4% of the model. The base weights stay frozen.
- **An adapter fits only the base it was trained on.** The catalogue
  identifies the base weights, and the extension offers adapters only from a
  catalogue whose base matches the one installed.
- **Room for new tasks is already in the vocabulary.** The tokenizer reserves
  64 command-line task tokens, 36 spare tokens, and mode, style and marker
  tokens, so a new adapter needs no embedding resize and no change to the base.
- **Every file is verified.** Each downloaded file is checked against the
  SHA-256 in `model.json` (a mismatch deletes it), and the model must be
  served over HTTPS.

### Planned

- More adapters for different use cases, such as legal or marketing writing,
  are in progress. Each will be added to the same base and offered in a list
  that users choose from.
- One repository per adapter in the KeelCore organization, with each adapter
  offered at more than one LoRA rank where there is time to train them.

---

## Runs in the browser

KeelCore is meant to run inside a web browser together with its adapters.
EchoMeBetter, the browser extension, runs the text-rewriting adapters.
Inference happens on your device, in a worker, through ONNX Runtime Web:

- **WebGPU** is the default wherever the browser offers a working GPU, with a
  **WebAssembly** (CPU) fallback. Processor, CPU usage and GPU power are
  settings.
- **Weight-only int8 ONNX export:** 0.76 GB on disk against 3.0 GB for the
  original weights, with 98.8% top-1 next-token agreement with fp32 and
  held-out loss 2.9201 → 2.9207.
- **Files are downloaded once** into the extension's private storage when the
  user asks; nothing model-sized ships in the extension package.
- **LoRA adapters in the browser.** ONNX Runtime Web had no way to load LoRA
  adapters, so we added it and contributed it upstream: a `LoraAdapter` API and
  an `activeLoraAdapters` run option for the standard (non-proxy) run path,
  merged into ONNX Runtime on 6 October 2026
  ([microsoft/onnxruntime#32886](https://github.com/microsoft/onnxruntime/pull/32886)).
  One adapter file can be reused across sessions, and it applies only to the
  run it is passed to.

---

## KeelCore at a glance

| | |
|---|---|
| **Architecture** | Encoder–decoder Transformer (sequence to sequence) |
| **Parameters** | 751,889,408 (tied input and output embeddings) |
| **Layers** | 24 encoder + 24 decoder |
| **Width** | d_model 1,024 · 16 heads × 64 · feed-forward 2,816 (gated GELU) |
| **Normalization** | RMSNorm, pre-norm |
| **Positions** | Relative position bias, 32 buckets, max distance 128 |
| **Context** | 2,048 tokens in, 2,048 tokens out |
| **Vocabulary** | 33,728 SentencePiece pieces: 32,200 learned + 896 command-line + 256 sentinels + control tokens |
| **Precision** | bf16 mixed precision, gradient checkpointing |
| **Dropout** | 0 |
| **Compatibility** | The pretrained checkpoint loads unchanged in Hugging Face Transformers (logits match the training code to 1e-5) and exports to ONNX |

Every dimension lives in [`model_config.yml`](model_config.yml). Nothing is
hardcoded in Python: to reconfigure the model, edit the config.

### Why an encoder–decoder

A bidirectional encoder reads the whole input before anything is generated, and
cross-attention lets the decoder look straight back at that input at every
step. A task that transforms text it was given can therefore copy a span
verbatim as a lookup instead of an act of memory, and keep URLs, code, numbers
and names exactly as written. At this size that wins over a decoder-only model
carrying the same information forward through its hidden state, and it is what
makes the model a good carrier for adapters.

---

## Tokenizer: English, code and the command line

One tokenizer serves every adapter, so it has to work for English prose, source
code and command-line text together. It is a SentencePiece unigram model with
byte fallback (no input is ever unrepresentable), identity normalization and
whitespace kept, so indentation, tabs, newlines, URLs and numbers (digits are
split one per piece) decode to exactly the text that went in. In a check of 500
documents from each of five datasets, every document round-tripped exactly.

| Special tokens | Pieces | Purpose |
|---|---:|---|
| Command-line vocabulary | 896 | Whole pieces of literal command-line text: 219 command names, 384 flags and 293 paths (for example `apt-get`, `--no-edit`, `/etc/nginx/`), so they are not split into fragments |
| Command-line task tokens | 64 | Reserved `<cli_reserved_N>` tokens for a command-line helper task, so it can be trained later without resizing the embedding table |
| Span sentinels | 256 | One per masked span in the denoising objectives |
| Mode and style tokens | 15 | Three objective modes and twelve style tokens |
| Markers and spares | 38 | Two structural markers and 36 spare tokens |

The command-line pieces are admitted by a rule, not by frequency alone. Text
that is also an English word or a word beginning, such as `cat`, is left out,
because a piece `cat` would cut `category` into `cat` and `egory` and make
English longer. 229 such candidates were rejected. What is admitted either
contains a character English words do not (a hyphen, slash, dot, underscore or
digit), is a capitalised file name such as `Dockerfile`, or is a command name
checked one by one against the prose corpus.

Measured against the earlier 32,832-piece version of the tokenizer, which lacks
the 896 pieces:

| Text | Tokens before | Tokens now |
|---|---:|---:|
| `ls -la /dev/null` | 7 | 3 |
| `chmod 755 /var/www/html/index.html` | 15 | 10 |
| `docker run --rm -it ubuntu` | 7 | 5 |
| `git commit --amend --no-edit` | 9 | 7 |
| `find . -name '*.log' -exec rm {} +` | 13 | 11 |
| `ssh-keygen -t rsa` | 7 | 6 |
| `apt-get install` | 4 | 4 |
| `sudo systemctl restart nginx.service` | 7 | 7 |

On sampled text, command-line text gets shorter (tldr pages 0.9% fewer tokens,
command cheat sheets 2.5% fewer) while English and code are essentially
unchanged (educational web 0.05%, general web 0.04%, source code 0.01% fewer).
Average characters per token: 4.3 on educational and general web text, 5.4 on
synthetic textbooks, 2.9 on source code, 3.7 on command-line pages.

---

## How it was pretrained

- **Objective: UL2 mixture of denoisers.** Every batch mixes three objectives
  so one model learns to fill in, reconstruct and continue text.

  | Mode | What it does | Sampling weight |
  |---|---|---:|
  | `[R]` regular denoising | Fills in short masked spans (about 15% of tokens, mean span 3) | 0.40 |
  | `[X]` extreme denoising | Rebuilds long or densely masked stretches (about 50% of tokens, mean span 32) | 0.35 |
  | `[S]` sequential | Reads the start of a document (25–75%) and continues it | 0.25 |

- **Data.** Seven public datasets covering educational and general web text,
  synthetic textbooks, source code, public-domain books and two small
  command-line references: 63.05 billion raw tokens under KeelCore's own
  tokenizer, 68.48 billion processed tokens once sentinels and objective
  formatting are added. Through step 81,000 each file is read once; the
  cooldown revisits the small command-line sets, capped at about eight passes
  each. The datasets and their token counts are listed in section 2 of the
  [technical report](https://saravanan-a-r.github.io/KeelCore/).
- **Optimizer.** AdamW, β = (0.9, 0.95), ε = 1e-8, weight decay 0.1, global
  gradient-norm clip 1.0. Peak learning rate 3e-4, with the position-bias
  tables at ×100 and the attention queries at ×0.1.
- **Schedule.** Warmup–stable–decay: warmup to step 900, constant to 81,000,
  linear decay to zero at 90,000.
- **Cooldown.** The decay phase (steps 81,000 to 90,000) also switched the data
  to a curated blend weighted toward educational text, code and command-line
  material, and one batch in five carried a synthetic exercise: damaged text in,
  the original out. It trains precise edits inside a faithful copy, and uses
  the reserved task-frame tokens that adapters can build on.
- **Batch.** An optimizer step closes once it holds at least 255,000 graded
  target tokens (mean 262,603 graded, 760,883 including encoder input).
- **Hardware.** One NVIDIA A100 40 GB, 40.5 days (15 Sep – 25 Oct 2026), peak
  memory about 27 GB, flat for the whole run.
- **Seeds.** Initialization 42, data order 1234. A restarted run continues
  bit-identically: data order, position and random state.

The run was instrumented end to end: every checkpoint went through CPU probes
while training continued on the GPU. The full history, including every issue
found and how it was handled, is in the [technical report](https://saravanan-a-r.github.io/KeelCore/).

---

## Results from the first pretraining run

### Training

| | |
|---|---:|
| Final held-out loss (master set, step 90,000) | 2.3217 (from 2.9456 at step 10,000) |
| Final perplexity | 10.2 (from 19.0) |
| Tokens processed | 68.5 billion |
| Optimizer steps applied | 90,000 of 90,000, none skipped, no NaN or Inf |
| Calibration error (ECE) | 0.009 |

Master evaluation (50,000 held-out records, scored every 10,000 steps; it alone
selects the best checkpoint; `[R]`, `[X]`, `[S]` are the per-objective losses):

| Step | Master loss | `[R]` | `[X]` | `[S]` |
|---:|---:|---:|---:|---:|
| 10,000 | 2.9456 | 2.137 | 3.038 | 3.152 |
| 20,000 | 2.8074 | 1.865 | 2.892 | 3.086 |
| 30,000 | 2.8152 | 1.909 | 2.895 | 3.085 |
| 40,000 | 2.5562 | 1.696 | 2.650 | 2.783 |
| 50,000 | 2.5070 | 1.629 | 2.623 | 2.706 |
| 60,000 | 2.4909 | 1.559 | 2.593 | 2.737 |
| 70,000 | 2.5099 | 1.686 | 2.653 | 2.642 |
| 80,000 | 2.4492 | 1.543 | 2.562 | 2.666 |
| 90,000 | **2.3217** | 1.413 | 2.478 | 2.470 |

The master loss rises slightly at 30,000 and 70,000 because the corpus is read
one source at a time and those checkpoints were saved inside a block of books
and a block of source code. Step 90,000, the end of the decay, is the best of
the run.

Held-out loss on unseen documents from each source at step 90,000 (nats per
token): educational web 2.2631, general web 2.6586, source code 1.0059,
textbooks 1.6175, books 2.7693. Every source but books is at its best on the
final checkpoint; books peaked at step 69,000 (2.7097).

### What the model learned

Capability probes on held-out text at step 90,000:

- **Word order.** Cosine similarity of encoder states before and after
  shuffling the input is 0.40 (1.0 would mean word order is ignored; it was
  0.93 at step 1,000).
- **Local context.** Scrambling the words beside a gap costs 4.87 extra nats,
  so predictions are grounded in nearby text.
- **Grounding in the input.** Giving the decoder another document's encoding
  costs 6.38 nats.
- **Token accuracy.** 54.8% top-1 when filling short gaps (`[R]`) and 45.6%
  when continuing text (`[S]`).
- **Calibration.** Expected calibration error 0.009: the model's confidence
  matches how often it is right.
- **Repetition.** Under plain greedy decoding, 14 of 32 held-out prompts fall
  into a repetition loop (30 of 32 at step 1,000). This is a property of base
  language models under greedy decoding and is reported as such.

### Reliability checks

| Check | Result |
|---|---|
| Steps applied | 90,000 of 90,000, no skipped updates, NaNs or Infs |
| Decoder causality | 0 leaks, verified on 83 checkpoints |
| Packed-document isolation | 0 leaks |
| Padding invariance | logits change by at most 5.1e-5, at every one of 83 checkpoints tested |
| KV-cache agreement | 100%: cached generation matches full recomputation |
| Loss reproduction | rebuilding the next training batch from a checkpoint reproduces the logged loss to within 1.9e-4 (20 checks) |
| Resume | bit-identical |
| Memory | flat for 40 days, peak about 27 GB of 40 GB |
| Checkpoint hash chain | 90 checkpoints chained, 0 broken links |

### Comparison with open models

KeelCore-752M, a base model with no instruction tuning, against 13 open models
from 355M to 1.3B parameters. Every model is evaluated with one harness, the
same questions and full test sets, zero-shot: each answer option is scored by
its log-likelihood, with the prompt wrapper chosen per model family. Where a
family has several formats (KeelCore: prefix, regular and extreme denoising;
T5 and Flan-T5: span or plain), the best format on each task is shown. The
average is the plain mean over the 15 benchmarks. **Every result is shown,
including those where KeelCore does not lead.**

#### Zero-shot accuracy, 15 benchmarks

| Rank | Model | Average accuracy (15 benchmarks) | Pretraining tokens |
|---|---|---:|---:|
| 1 | flan-t5-large | 56.9 | ~1T + instruction tuning |
| 2 | smollm2-360m | 54.0 | 4T |
| 3 | qwen2.5-0.5b | 52.9 | 18T |
| 4 | opt-1.3b | 50.1 | 180B |
| 5 | pythia-1b | 47.6 | 300B |
| 6 | t5-large | 46.6 | ~1T |
| 7 | gpt2-large | 46.4 | not stated |
| 8 | bart-large | 46.1 | not stated |
| 9 | pythia-410m | 45.3 | 300B |
| 10 | KeelCore | 45.3 | 69B |
| 11 | t5-v1_1-large | 44.7 | ~1T |
| 12 | gpt2-medium | 44.0 | not stated |
| 13 | opt-350m | 42.8 | 180B |
| 14 | bloom-560m | 41.6 | 366B |

KeelCore ranks **10 of 14** by average accuracy (45.3). It is ahead of
t5-v1_1-large, gpt2-medium, opt-350m and bloom-560m, and behind flan-t5-large,
smollm2-360m, qwen2.5-0.5b, opt-1.3b, pythia-1b, t5-large, gpt2-large,
bart-large and pythia-410m. It gets there on a 69B-token budget: the models
that score clearly higher saw 2.6× to 261× as many tokens, and KeelCore is
level with models trained on 4.3× and 14× more. The results are promising for
a first run at this budget.

Where KeelCore ranks on each benchmark:

| Benchmark | KeelCore rank (of 14) | KeelCore score | Best model | Best score | Chance |
|---|---:|---:|---:|---:|---:|
| ARC-Easy | 6 | 56.1 | smollm2-360m | 70.4 | 25 |
| ARC-Chall. | 7 | 26.6 | smollm2-360m | 38.7 | 25 |
| BoolQ | 13 | 55.3 | flan-t5-large | 86.3 | 50 |
| CSQA | 13 | 30.9 | flan-t5-large | 54.6 | 20 |
| COPA | 12 | 62.6 | flan-t5-large | 73.6 | 50 |
| HellaSwag | 7 | 44.1 | smollm2-360m | 56.4 | 25 |
| LAMBADA | 13 | 40.4 | flan-t5-large | 67.0 | 0 |
| MMLU | 4 | 25.5 | qwen2.5-0.5b | 47.7 | 25 |
| OpenBookQA | 7 | 31.0 | smollm2-360m | 37.4 | 25 |
| PIQA | 8 | 66.4 | flan-t5-large | 72.4 | 50 |
| QASC | 10 | 27.2 | smollm2-360m | 47.9 | 12.5 |
| RACE-high | 8 | 32.6 | flan-t5-large | 41.4 | 25 |
| SciQ | 6 | 88.4 | flan-t5-large | 93.3 | 25 |
| SocialIQA | 8 | 39.7 | flan-t5-large | 50.1 | 33.3 |
| Winogrande | 14 | 52.2 | flan-t5-large | 62.7 | 50 |

With 69B pretraining tokens, KeelCore is in the top ten of 14 on ten of the 15
benchmarks (ARC-Easy, ARC-Challenge, HellaSwag, MMLU, OpenBookQA, PIQA, QASC,
RACE-high, SciQ, SocialIQA), and its best result is 4th of 14 on MMLU. It
ranks below 10th on five (BoolQ, CommonsenseQA, COPA, LAMBADA, Winogrande). Several tasks sit close to
chance for every model of this size, so small gaps there are within noise.

<details>
<summary>Every model, every benchmark</summary>

| Model | ARC-Easy | ARC-Chall. | BoolQ | CSQA | COPA | HellaSwag | LAMBADA | MMLU | OpenBookQA | PIQA | QASC | RACE-high | SciQ | SocialIQA | Winogrande | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flan-t5-large | 56.9 | 30.4 | 86.3 | 54.6 | 73.6 | 48.8 | 67.0 | 41.5 | 35.2 | 72.4 | 39.5 | 41.4 | 93.3 | 50.1 | 62.7 | 56.9 |
| smollm2-360m | 70.4 | 38.7 | 62.0 | 45.7 | 73.4 | 56.4 | 53.8 | 26.3 | 37.4 | 72.0 | 47.9 | 34.4 | 91.1 | 40.8 | 59.0 | 54.0 |
| qwen2.5-0.5b | 64.6 | 32.3 | 61.8 | 41.6 | 72.0 | 52.2 | 52.5 | 47.7 | 35.2 | 70.1 | 33.7 | 35.6 | 93.1 | 44.2 | 56.4 | 52.9 |
| opt-1.3b | 56.7 | 27.7 | 58.2 | 42.0 | 73.0 | 53.7 | 58.8 | 25.1 | 32.4 | 70.9 | 34.4 | 33.0 | 85.3 | 40.3 | 59.4 | 50.1 |
| pythia-1b | 57.0 | 26.9 | 60.3 | 35.5 | 69.2 | 47.3 | 56.3 | 23.1 | 31.2 | 69.4 | 29.7 | 31.5 | 84.0 | 39.9 | 53.4 | 47.6 |
| t5-large | 43.7 | 25.6 | 68.1 | 41.8 | 66.4 | 40.2 | 52.8 | 25.2 | 32.4 | 57.7 | 26.2 | 33.6 | 89.2 | 39.3 | 56.5 | 46.6 |
| gpt2-large | 53.2 | 25.0 | 60.9 | 36.3 | 70.0 | 45.3 | 47.7 | 23.2 | 29.0 | 69.3 | 29.8 | 30.7 | 80.4 | 39.6 | 55.3 | 46.4 |
| bart-large | 45.5 | 22.2 | 63.1 | 36.1 | 68.0 | 36.7 | 61.6 | 23.1 | 27.0 | 60.9 | 31.0 | 32.7 | 86.8 | 40.2 | 55.8 | 46.1 |
| pythia-410m | 51.9 | 24.4 | 60.2 | 32.8 | 67.2 | 40.5 | 51.6 | 23.1 | 29.4 | 67.1 | 28.2 | 29.3 | 81.2 | 38.9 | 53.3 | 45.3 |
| **KeelCore** | **56.1** | **26.6** | **55.3** | **30.9** | **62.6** | **44.1** | **40.4** | **25.5** | **31.0** | **66.4** | **27.2** | **32.6** | **88.4** | **39.7** | **52.2** | **45.3** |
| t5-v1_1-large | 44.3 | 27.0 | 68.3 | 39.6 | 53.4 | 28.6 | 58.1 | 24.4 | 28.4 | 52.9 | 26.6 | 35.7 | 89.3 | 38.9 | 54.4 | 44.7 |
| gpt2-medium | 49.1 | 25.2 | 59.0 | 32.6 | 65.4 | 39.4 | 43.0 | 22.9 | 29.4 | 66.4 | 27.4 | 30.4 | 77.1 | 39.1 | 53.1 | 44.0 |
| opt-350m | 45.4 | 23.6 | 56.7 | 31.5 | 65.6 | 36.9 | 47.0 | 23.1 | 27.4 | 64.4 | 22.7 | 28.4 | 76.7 | 39.9 | 52.3 | 42.8 |
| bloom-560m | 47.3 | 23.6 | 55.2 | 26.4 | 60.8 | 36.6 | 35.4 | 23.0 | 26.8 | 65.1 | 23.2 | 29.3 | 81.1 | 37.1 | 52.9 | 41.6 |

</details>

#### Score for the training budget

KeelCore saw 69B tokens in pretraining. This table lists every model whose
budget is stated, from smallest to largest, with the budget as a multiple of
KeelCore's and its average accuracy against KeelCore's (45.3). Budgets are as
stated in each model's published description (approximate); GPT-2 and BART do
not state a token count.

| Model | Pretraining tokens | Tokens vs KeelCore | Avg accuracy | vs KeelCore |
|---|---:|---:|---:|---:|
| **KeelCore** | **69B** | **1x** | **45.3** | **+0.0** |
| opt-1.3b | 180B | 2.6x | 50.1 | +4.8 |
| opt-350m | 180B | 2.6x | 42.8 | -2.5 |
| pythia-1b | 300B | 4.3x | 47.6 | +2.4 |
| pythia-410m | 300B | 4.3x | 45.3 | +0.0 |
| bloom-560m | 366B | 5.3x | 41.6 | -3.7 |
| t5-v1_1-large | ~1T | 14x | 44.7 | -0.6 |
| flan-t5-large | ~1T + instruction tuning | 14x | 56.9 | +11.7 |
| t5-large | ~1T | 14x | 46.6 | +1.3 |
| smollm2-360m | 4T | 58x | 54.0 | +8.7 |
| qwen2.5-0.5b | 18T | 261x | 52.9 | +7.6 |

- **Level with models trained on several times more.** KeelCore is within 1.5
  points of pythia-410m (4.3× the tokens) and t5-large (14× the tokens), and
  ahead of opt-350m (2.6×), bloom-560m (5.3×) and t5-v1_1-large (14×), all of
  which saw more tokens.
- **Every clearly stronger model had a larger budget.** The models more than
  1.5 points ahead saw 2.6× (opt-1.3b), 4.3× (pythia-1b), 14× (flan-t5-large),
  58× (smollm2-360m) and 261× (qwen2.5-0.5b) as many tokens as KeelCore. The two
  closest in budget, opt-1.3b and pythia-1b, are also larger models (about 1.3B
  and 1B parameters), and flan-t5-large was additionally instruction-tuned.
- **Nothing here points to the architecture as the limit.** Two other
  encoder–decoder models, trained on about 14× the tokens, bracket KeelCore
  (44.7 and 46.6 against 45.3). KeelCore's held-out loss was still improving at
  the end of the stable phase (master loss 2.56 at step 40,000 and 2.45 at step
  80,000, before the decay began), so the run ended because the token budget
  did.
- **What this does not show.** KeelCore was trained once, at one budget. That a
  larger budget would close the gap is a reading of these results, and testing
  it needs a longer run.

#### Prediction on fresh text

Each model predicts the second half of documents published in August–October
2026, after every model's training data was collected: 483 Wikipedia articles,
381 arXiv abstracts and 377 source files. Bits per byte, lower is better,
ordered by the mean; bits per byte is comparable across tokenizers.
KeelCore ranks 8 of 14 on Wikipedia, 8 of 14 on arXiv and 6 of 14 on code.

| Model | Wikipedia | arXiv | Code | Mean |
|---|---:|---:|---:|---:|
| qwen2.5-0.5b | 0.815 | 0.899 | 0.686 | 0.800 |
| smollm2-360m | 0.799 | 0.913 | 0.732 | 0.815 |
| pythia-1b | 0.786 | 0.888 | 0.770 | 0.815 |
| pythia-410m | 0.836 | 0.931 | 0.823 | 0.863 |
| bloom-560m | 0.926 | 0.976 | 0.894 | 0.932 |
| opt-1.3b | 0.770 | 0.965 | 1.133 | 0.956 |
| **KeelCore** | **0.883** | **1.002** | **1.086** | **0.990** |
| gpt2-medium | 0.878 | 1.032 | 1.340 | 1.084 |
| gpt2-large | 0.837 | 0.998 | 1.527 | 1.120 |
| opt-350m | 0.890 | 1.088 | 1.385 | 1.121 |
| flan-t5-large | 0.934 | 1.089 | 1.433 | 1.152 |
| bart-large | 1.586 | 1.761 | 3.602 | 2.316 |
| t5-large | 4.285 | 3.422 | 6.669 | 4.792 |
| t5-v1_1-large | 4.505 | 4.100 | 7.454 | 5.353 |

The full methodology, every figure and the complete training history are in the
[technical report](https://saravanan-a-r.github.io/KeelCore/).

---

## Repository layout

```
tokenizer/training/  tokenizer construction, 11 acceptance gates and the freeze
                     protocol (byte-fallback subword model; the frozen
                     tokenizer and its checksums live in output/)
pretrain_corpus/     downloading and cleaning the pretraining data: clean, PII
                     masking, English-prose quality filter, exact and
                     normalized deduplication
corpus/              the resumable pretraining reader: text to token IDs, plus
                     the blended reader used for the cooldown
UL2/                 the mixture-of-denoisers pretraining objective
model/               the encoder–decoder itself, with Hugging Face export
rewrite/             the synthetic damaged-text task mixed into the cooldown
training/            the training loop, optimizer, schedules, checkpointing
monitoring/          the TensorBoard dashboard, per-checkpoint probes, GPU and
                     optimizer statistics
ownership/           checkpoint hash chain, fingerprint and embedding signature
chrome_extension/    the EchoMeBetter browser extension: downloads the base and
                     the text-rewriting adapters, and runs them on-device
                     with ONNX Runtime Web
github_site/         the technical report, published with GitHub Pages
test/                end-to-end pipeline tests across the packages
model_config.yml     the model's shape: one place to change it
training_config.yml  pretraining stages and hyperparameters
ownership_config.yml ownership settings
pretrain.py          entrypoint wiring the packages into a real pretraining run
pretrain.sh          the full run, with its dashboard
pretrain_smoke_test.py  pre-flight check before a rehearsal or full run
```

Each package has its own test suite and a configuration file that is the single
source of truth for its parameters. No dimension or hyperparameter is
hardcoded in Python.

## Development

Python 3.12 and PyTorch 2.6 were used for the training stack. Run each
package's tests from inside its own directory:

```bash
cd model && python -m pytest             # add -m "not slow" to skip full-size models
cd UL2 && python -m pytest
cd training && python -m pytest
python -m pytest test                     # end-to-end, from the repository root
```

Rehearse a stage before committing compute to it:

```bash
python pretrain_smoke_test.py --stage pretrain --device cuda \
    --corpus pretrain_corpus/output/pretrain_output \
    --quick-eval-corpus  pretrain_corpus/output/evals/quick_eval.jsonl \
    --master-eval-corpus pretrain_corpus/output/evals/master_eval.jsonl
```

The extension is built from `chrome_extension/` (`npm run build`, `npm test`,
`npm run typecheck`). It depends on a custom onnxruntime-web build with LoRA
adapter support, installed from `chrome_extension/vendor/`, which is kept out
of git because of its size. Model files are not bundled: the extension
downloads them from the folder named in `chrome_extension/model.source.json`.

---

## Contact

For collaboration or anything else:

- **Email:** sararavi.ar@gmail.com
- **GitHub:** [@saravanan-a-r](https://github.com/saravanan-a-r)
- **LinkedIn:** [saravanan-a-r](https://www.linkedin.com/in/saravanan-a-r/)
