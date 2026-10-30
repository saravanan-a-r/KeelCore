/**
 * The whole rewrite path for the loaded base model: prompt → encoder →
 * greedy decoding → text, with the style's adapter active. Runtime-agnostic:
 * it is handed ready sessions, adapters and a Tensor constructor, so it runs
 * the same under onnxruntime-web in the extension's worker and under Node in
 * the test suite.
 *
 * The base (sessions, tokenizer) stays loaded between rewrites. Everything
 * per style comes with each request: the adapter to run with and the
 * manifest that describes it, which may be newer than the one the base was
 * loaded from (adapters are added to an installed base over time).
 */
import { EchoError } from '../shared/errors';
import type { StyleId } from '../shared/styles';
import { greedyGenerate, type GreedyResult } from './generation/greedyDecoder';
import type { ModelManifest } from './manifest';
import { T5Runner, type AdapterPair, type T5Geometry, type TensorFactory } from './onnx/t5Runner';
import { PromptBuilder } from './prompt/promptBuilder';
import { applyInputTransforms, undoInputTransforms } from './prompt/textTransforms';
import { Tokenizer } from './tokenizer/tokenizer';
import type { InferenceSession } from 'onnxruntime-web';

/**
 * Every style chunks long input the same way; only the expansion ratio
 * differs, because each adapter was trained with its own target/input
 * length bound (SFT_adapter_rephrase/adapter_config.yml `styles.<style>`,
 * or the style-neutral default for a style without its own block). A
 * chunk's raw text is kept small enough that even a full-ratio rewrite of
 * it still fits `limits.maxNewTokens`, so splitting removes INPUT_TOO_LONG
 * and OUTPUT_TOO_LONG instead of trading one for the other.
 */
const MAX_EXPANSION: Record<StyleId, number> = {
  professional: 1.6, // styles.professional.max_length_ratio
  concise: 1.15, // styles.concise.max_length_ratio (concise must not expand)
  elaborate: 3.0, // no styles.elaborate block: the shared data.max_length_ratio default
  friendly: 3.0, // no styles.friendly block: the shared data.max_length_ratio default
  grammar: 3.0, // no styles.grammar block: the shared data.max_length_ratio default
};

/** A run of text that goes through the model, or a separator carried through untouched. */
type Segment = { readonly kind: 'chunk'; readonly text: string } | { readonly kind: 'join'; readonly text: string };

const PARAGRAPH_SPLIT = /(\n{2,})/;
const SENTENCE_SPLIT = /(?<=[.!?])\s+/;
const WORD_SPLIT = /\s+/;

export interface RewriteOptions {
  readonly signal?: AbortSignal;
  readonly onProgress?: (generatedTokens: number) => void;
  /**
   * After each generated token, a way to read the rewrite so far. Reading
   * decodes, so a caller showing it reads only as often as it shows it.
   * The text so far is not the result: rewrite() resolves with that.
   */
  readonly onPartial?: (readText: () => string) => void;
}

export interface RewriteResult {
  readonly text: string;
  readonly inputTokens: number;
  readonly outputTokens: number;
}

export interface RewriteRequest {
  /** The installed manifest: prompt frame, limits and decoding settings. */
  readonly manifest: ModelManifest;
  /** The adapter the style runs with (its own, or the fallback). */
  readonly adapterId: StyleId;
  readonly adapter: AdapterPair;
  readonly text: string;
}

export interface EngineParts {
  readonly geometry: T5Geometry;
  readonly tokenizerJson: unknown;
  readonly encoder: InferenceSession;
  readonly decoder: InferenceSession;
  readonly TensorCtor: TensorFactory;
}

export class RewriteEngine {
  private readonly tokenizer: Tokenizer;
  private readonly runner: T5Runner;
  private readonly sessions: readonly InferenceSession[];
  private readonly prompts = new WeakMap<ModelManifest, PromptBuilder>();

  constructor(parts: EngineParts) {
    this.sessions = [parts.encoder, parts.decoder];
    this.tokenizer = Tokenizer.fromJson(parts.tokenizerJson);
    this.runner = new T5Runner(parts.TensorCtor, parts.encoder, parts.decoder, parts.geometry);
  }

  /** Encoder input IDs for a request; exposed so tests can check them against the reference. */
  promptIds(manifest: ModelManifest, adapterId: StyleId, text: string): number[] {
    let prompts = this.prompts.get(manifest);
    if (!prompts) {
      prompts = new PromptBuilder(manifest, this.tokenizer);
      this.prompts.set(manifest, prompts);
    }
    return prompts.build(adapterId, text);
  }

  /** Greedy generation with `adapter` active; null runs the plain base model. */
  generateIds(manifest: ModelManifest, inputIds: readonly number[], adapter: AdapterPair | null, options: RewriteOptions = {}): Promise<GreedyResult> {
    const { tokens, generation, limits } = manifest;
    return greedyGenerate(adapter ? this.runner.withAdapter(adapter) : this.runner, inputIds, {
      ...generation,
      decoderStartId: tokens.decoderStartId,
      eosId: tokens.eosId,
      maxNewTokens: limits.maxNewTokens,
      signal: options.signal,
      onToken: (count, generated) => {
        options.onProgress?.(count);
        options.onPartial?.(() => this.partialText(manifest, generated));
      },
    });
  }

  /** Generated IDs → text, as the training codec decodes: padding skipped, input transforms undone. */
  decode(manifest: ModelManifest, ids: readonly number[]): string {
    const text = this.tokenizer.decode(ids.filter((id) => id !== manifest.tokens.padId));
    return undoInputTransforms(text, manifest.inputTransforms);
  }

  /**
   * The rewrite so far, for showing while it is written. A character spelled
   * in several byte tokens (an emoji, many accented letters) is incomplete
   * until its last byte arrives and decodes as U+FFFD meanwhile: those are
   * left off the end until it is whole.
   */
  private partialText(manifest: ModelManifest, ids: readonly number[]): string {
    return this.decode(manifest, ids).replace(/\uFFFD+$/u, '').trim();
  }

  /** Free the sessions' memory; the engine cannot be used afterwards. */
  async release(): Promise<void> {
    await Promise.all(this.sessions.map((session) => session.release()));
  }

  /**
   * Identical to the pre-chunking engine for every call that already
   * succeeded: only a request that would have failed (input over the
   * frame's limit, or a single shot running past maxNewTokens) takes the
   * chunked path, so nothing that worked before changes.
   */
  async rewrite(request: RewriteRequest, options: RewriteOptions = {}): Promise<RewriteResult> {
    const { inputLimit, chunkBudget } = this.chunkBudgets(request.manifest, request.adapterId);
    if (this.measureTokens(request.manifest, request.text) > inputLimit) {
      return this.rewriteChunked(request, options, chunkBudget);
    }
    try {
      return await this.rewriteOnce(request, options);
    } catch (error) {
      if (error instanceof EchoError && error.code === 'OUTPUT_TOO_LONG') {
        return this.rewriteChunked(request, options, chunkBudget);
      }
      throw error;
    }
  }

  /** The original one-shot path: exactly what every style ran before chunking existed. */
  private async rewriteOnce({ manifest, adapterId, adapter, text }: RewriteRequest, options: RewriteOptions): Promise<RewriteResult> {
    const inputIds = this.promptIds(manifest, adapterId, text);
    const result = await this.generateIds(manifest, inputIds, adapter, options);
    if (!result.finished) throw new EchoError('OUTPUT_TOO_LONG', { limit: manifest.limits.maxNewTokens });
    const output = this.decode(manifest, result.ids).trim();
    if (output.length === 0) throw new EchoError('EMPTY_RESULT');
    return { text: output, inputTokens: inputIds.length, outputTokens: result.ids.length };
  }

  /** Tokens `text` alone would add to the prompt, exactly as the frame would encode it. */
  private measureTokens(manifest: ModelManifest, text: string): number {
    if (text.length === 0) return 0;
    return this.tokenizer.encode(applyInputTransforms(text, manifest.inputTransforms)).length;
  }

  /**
   * `inputLimit`: the raw-text budget the frame has room for -- exceeding
   * this is what `promptIds` itself would refuse with INPUT_TOO_LONG.
   * `chunkBudget`: the smaller, output-ratio-aware size actually used once
   * chunking has been decided on, so a chunk's rewrite also fits
   * `maxNewTokens`. Frame overhead is measured from the manifest itself (an
   * empty prompt build), so it tracks the real frame, not a hardcoded guess.
   */
  private chunkBudgets(manifest: ModelManifest, adapterId: StyleId): { inputLimit: number; chunkBudget: number } {
    const overhead = this.promptIds(manifest, adapterId, '').length;
    const inputLimit = Math.max(1, manifest.limits.maxInputTokens - overhead);
    const outputBudget = Math.max(1, Math.floor(manifest.limits.maxNewTokens / MAX_EXPANSION[adapterId]));
    return { inputLimit, chunkBudget: Math.min(inputLimit, outputBudget) };
  }

  /** Pack `pieces` (already in order) into as few `joiner`-joined strings as fit under `budget` tokens each. */
  private packSegments(manifest: ModelManifest, pieces: readonly string[], budget: number, joiner: string): string[] {
    const packed: string[] = [];
    let current = '';
    for (const piece of pieces) {
      const candidate = current ? `${current}${joiner}${piece}` : piece;
      if (this.measureTokens(manifest, candidate) <= budget) {
        current = candidate;
        continue;
      }
      if (current) packed.push(current);
      if (this.measureTokens(manifest, piece) <= budget) {
        current = piece;
      } else {
        packed.push(...this.hardFragment(manifest, piece, budget));
        current = '';
      }
    }
    if (current) packed.push(current);
    return packed;
  }

  /** A single sentence (or paragraph) too big on its own: fall back to word groups, or fail with an error. */
  private hardFragment(manifest: ModelManifest, text: string, budget: number): string[] {
    const words = text.split(WORD_SPLIT).filter((word) => word.length > 0);
    if (words.length <= 1) {
      throw new EchoError('INPUT_TOO_LONG', { actual: this.measureTokens(manifest, text), limit: budget });
    }
    return this.packSegments(manifest, words, budget, ' ');
  }

  /** The whole input, as an ordered list of model calls and the text to carry through untouched between them. */
  private planSegments(manifest: ModelManifest, text: string, budget: number): Segment[] {
    const segments: Segment[] = [];
    const paragraphParts = text.split(PARAGRAPH_SPLIT);
    for (let index = 0; index < paragraphParts.length; index += 2) {
      const paragraph = paragraphParts[index]!;
      const separator = paragraphParts[index + 1];
      const sentences = paragraph.split(SENTENCE_SPLIT).filter((sentence) => sentence.length > 0);
      const chunks = this.packSegments(manifest, sentences, budget, ' ');
      chunks.forEach((chunk, chunkIndex) => {
        if (chunkIndex > 0) segments.push({ kind: 'join', text: ' ' });
        segments.push({ kind: 'chunk', text: chunk });
      });
      if (separator !== undefined) segments.push({ kind: 'join', text: separator });
    }
    return segments;
  }

  /** Rewrite each chunk in order, streaming the joined text so far through `onPartial`. */
  private async rewriteChunked(request: RewriteRequest, options: RewriteOptions, budget: number): Promise<RewriteResult> {
    const { manifest, adapterId, adapter, text } = request;
    const segments = this.planSegments(manifest, text, budget);
    // Build every chunk's prompt before the first model call, so text the prompt builder refuses (reserved
    // markup) fails at once, as it does for a short text, instead of after the earlier chunks were rewritten.
    for (const segment of segments) {
      if (segment.kind === 'chunk') this.promptIds(manifest, adapterId, segment.text);
    }
    let completed = '';
    let inputTokens = 0;
    let outputTokens = 0;
    for (const segment of segments) {
      if (options.signal?.aborted) throw new EchoError('CANCELLED');
      if (segment.kind === 'join') {
        completed += segment.text;
        continue;
      }
      const prefix = completed;
      const result = await this.rewriteOnce(
        { manifest, adapterId, adapter, text: segment.text },
        {
          signal: options.signal,
          onProgress: options.onProgress,
          onPartial: options.onPartial ? (readText) => options.onPartial!(() => prefix + readText()) : undefined,
        },
      );
      completed = prefix + result.text;
      inputTokens += result.inputTokens;
      outputTokens += result.outputTokens;
    }
    const output = completed.trim();
    if (output.length === 0) throw new EchoError('EMPTY_RESULT');
    return { text: output, inputTokens, outputTokens };
  }
}
