# EchoMeBetter

A small, controllable **text rewriting model**, trained from scratch.

Given a piece of text and a requested style, it produces a single rewrite that
means the same thing — while preserving URLs, code, numbers, names and Markdown
exactly as they were written.

```
input   ·  <style:professional>
           "hey so the api at https://api.example.com/v2/users is broken again,
            returns 503 like 40% of the time. can u look?"

output  ·  "The API at https://api.example.com/v2/users is experiencing repeated
            failures, returning 503 errors in approximately 40% of requests.
            Could you investigate?"
```

The URL and both numbers survive character-for-character. That fidelity
requirement — not fluency — is what drives most of the design decisions here.

---

## Goal

Build a compact rewriting model that is small enough to serve cheaply (and to
quantize onto a single GPU or a laptop), while staying close to a much larger
teacher model on this one narrow task.

Rewriting is deliberately scoped: the model never answers questions, never adds
or removes information, never explains itself, and never returns more than one
version. It transforms — nothing else.

**Styles.** The first release trains five: `grammar`, `concise`, `elaborate`,
`professional`, `friendly`. A further set is planned; adding one is a data task,
not an architecture change.

---

## Architecture, at a glance

An **encoder–decoder (seq2seq) Transformer**, not a decoder-only LM.

The reasoning is task-specific: a bidirectional encoder reads the entire source
before anything is generated, and cross-attention lets the decoder look directly
back at that source at every step. Copying a URL correctly is fundamentally a
lookup, and cross-attention is a lookup mechanism. At this size, that wins over a
decoder-only model carrying the same information forward through its hidden
state.

```
                        ┌──────────────────────────┐
   style token +        │        ENCODER           │
   source text     ───► │   (bidirectional)        │
                        └────────────┬─────────────┘
                                     │  cross-attention
                        ┌────────────▼─────────────┐
                        │        DECODER           │  ───►  rewritten text
                        │   (autoregressive)       │
                        └──────────────────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │  preservation validator  │  ───►  repair / regenerate
                        └──────────────────────────┘
```

**Style control** is a single reserved vocabulary token prepended to the encoder
input. One token, one behaviour switch — no natural-language instruction to
misparse. All styles share one multi-task model.

**Preservation validator.** A deterministic post-generation check that verifies
URLs, emails, numbers and code survived intact, and that exactly one complete
rewrite was produced. On failure it repairs or regenerates. This is what turns a
probabilistic model into a dependable one, and it is not optional.

---

## Training stages

| Stage | What happens |
|---|---|
| **1 · Pretrain** | Mixture-of-denoisers (UL2-style) over a large commercial-safe English corpus. Teaches language, structure, and how code and URLs are shaped — before the model ever sees the rewriting task. |
| **2 · Supervised fine-tune** | Multi-task training on `(style, source) → rewrite` pairs distilled from a large teacher model. All styles train together. |
| **3 · Quality lift** | Filter the teacher's own mistakes out of the data, then preference-train the model to follow the hard constraints — preserve the link, emit exactly one output, don't answer the question. |

---

## Repository layout

```
tokenizer/    vocabulary construction, validation and freeze protocol
              (byte-fallback subword model — no input is ever unrepresentable)
corpus/       the resumable pretraining data pipeline (text -> token IDs)
UL2/          the mixture-of-denoisers pretraining objective
model/        the encoder–decoder itself
training/     the training loop, optimizer, schedules and checkpointing
rewrite/      the synthetic rewrite task mixed into the end of pretraining
              (corrupted text in, original out — the shape SFT starts from)
pretrain.py            entrypoint wiring the five packages into a real run
pretrain_smoke_test.py pre-flight check before a real rehearsal or full run
```

Each module has its own README, its own test suite, and a configuration file
that is the single source of truth for its parameters. No dimension or
hyperparameter is hardcoded in Python — to reconfigure, edit the config.

---

## Design principles

- **Fidelity beats compression.** Where a choice trades exactness for a shorter
  sequence or a faster path, exactness wins.
- **Fail loudly, never silently.** Silent truncation and silently dropped data
  are the characteristic failure modes of a pipeline like this. Every stage
  asserts.
- **Reserve now, use later.** Unused capacity costs a rounding error. Missing
  capacity costs a full retrain.
- **Frozen means frozen.** Decisions welded into every checkpoint's weight
  shapes are marked as such and are not revisited casually.

---

## Status

Tokenizer, model, pretraining objective, training loop and the corpus
pipeline are implemented and tested; `pretrain.py` wires them into a runnable
Stage 1 pipeline, and `pretrain_smoke_test.py` proves it end to end (§7.8)
before a real corpus is downloaded. Stage 1 pretraining is live on the full
corpus.

This pretrained model is designed as a shared foundation, not a single-purpose
model. The vocabulary and pretraining mix were chosen deliberately to serve
multiple downstream use cases as lightweight **LoRA adapters** trained on top
of one frozen base — rather than pretraining a separate model per use case.
Supervised fine-tuning of the first adapters is in progress.

---

## License

Licensed under the **[EchoMeBetter License, Version 1.0](LICENSE)** (Personal
and Research Use). All rights reserved.

In short:

- **Individuals** may use EchoMeBetter for their own writing, including
  writing for their own job or studies, **only through the official
  EchoMeBetter browser extension**.
- **Noncommercial research** is allowed, through the extension, and every
  publication or presentation must **cite this project** (see below).
- The model and code may not be extracted, modified, shared, sold, hosted,
  run outside the extension, or used to build or train other models.
- **Commercial use**, use by or for an organization, and **purchase of the
  model** need a separate written agreement. See
  [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

This summary is for convenience; the [LICENSE](LICENSE) file controls.

## Citation

Research that uses EchoMeBetter must cite it in every publication or
presentation (Section 4 of the [license](LICENSE)). Use one of the forms in
[NOTICE](NOTICE):

```
EchoMeBetter (2026) by Saravanan A R.
https://github.com/saravanan-a-r/EchoMeBetter
```

## Contact

For commercial licensing, purchase of the model (see
[COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md)), collaboration, or anything
else:

- **Email:** sararavi.ar@gmail.com
- **GitHub:** [@saravanan-a-r](https://github.com/saravanan-a-r)
- **LinkedIn:** [saravanan-a-r](https://www.linkedin.com/in/saravanan-a-r/)
