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
import { CrossKvRunner, type AdapterPair, type CrossKvGeometry, type TensorFactory } from './onnx/crossKvRunner';
import { PromptBuilder } from './prompt/promptBuilder';
import { applyInputTransforms, undoInputTransforms } from './prompt/textTransforms';
import { Tokenizer } from './tokenizer/tokenizer';
import type { InferenceSession } from 'onnxruntime-web';

/**
 * The most output tokens a style writes per input token, so a text goes
 * through one pass only when its rewrite fits `limits.maxNewTokens`; a longer
 * text is cut into pieces before any pass, and no pass is spent on an output
 * that cannot finish. Token ratios (target/input, the real tokenizer) on
 * 20,000 training pairs per style, for inputs of 100+ tokens (the only ones
 * near a budget): max professional 1.29, concise 1.16, elaborate 1.88,
 * friendly 1.45, grammar 1.24. Professional and concise keep their training
 * bound (SFT_adapter_rephrase/adapter_config.yml `styles.<style>.data`);
 * the others sit above that max and above the all-length p99.5 (2.12, 1.60,
 * 1.25). The released int8 model on long eval inputs stayed under 1.2 for
 * every style. Every budget (384 / ratio: 240, 333, 174, 225, 295) is under
 * the 512-token training source limit.
 */
export const MAX_EXPANSION: Readonly<Record<StyleId, number>> = {
  professional: 1.6, // styles.professional.data.max_length_ratio
  concise: 1.15, // styles.concise.data.max_length_ratio (concise must not expand)
  elaborate: 2.2,
  friendly: 1.7,
  grammar: 1.3,
};

/**
 * A run of text that goes through the model, or a separator carried through
 * untouched. `halved`: the chunk is half of one whose rewrite ran past
 * maxNewTokens; it is not cut again, so a model that never stops costs two
 * passes, not one per word.
 */
type Segment = { readonly kind: 'chunk'; readonly text: string; readonly halved?: boolean } | { readonly kind: 'join'; readonly text: string };

/**
 * How a text too long for one call is cut: paragraphs, then sentences, then
 * words. Each split keeps its separator, so the original line breaks and
 * spacing come back unchanged between the rewritten pieces.
 */
const SPLITS: readonly RegExp[] = [/(\n{2,})/, /((?<=[.!?])\s+)/, /(\s+)/];

export interface RewriteOptions {
  readonly signal?: AbortSignal;
  readonly onProgress?: (generatedTokens: number) => void;
  /**
   * After each generated token, a way to read the rewrite so far. Reading
   * decodes, so a caller showing it reads only as often as it shows it.
   * The text so far is not the result: rewrite() resolves with that.
   */
  readonly onPartial?: (readText: () => string) => void;
  /**
   * The text so far was dropped and the rewrite starts over in pieces: a
   * single pass ran past maxNewTokens. Text read through `onPartial` before
   * this is not part of the result.
   */
  readonly onRestart?: () => void;
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
  readonly geometry: CrossKvGeometry;
  readonly tokenizerJson: unknown;
  readonly encoder: InferenceSession;
  readonly decoder: InferenceSession;
  readonly TensorCtor: TensorFactory;
}

export class RewriteEngine {
  private readonly tokenizer: Tokenizer;
  private readonly runner: CrossKvRunner;
  private readonly sessions: readonly InferenceSession[];
  private readonly prompts = new WeakMap<ModelManifest, PromptBuilder>();

  constructor(parts: EngineParts) {
    this.sessions = [parts.encoder, parts.decoder];
    this.tokenizer = Tokenizer.fromJson(parts.tokenizerJson);
    this.runner = new CrossKvRunner(parts.TensorCtor, parts.encoder, parts.decoder, parts.geometry);
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
      piece: (id) => this.tokenizer.piece(id),
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
   * One pass when the whole rewrite fits `maxNewTokens` at the style's
   * expansion ratio, otherwise pieces from the start, so no pass is spent on
   * an output that cannot finish. A pass that still runs past the limit (an
   * output longer than the ratio allows) is redone once, in two halves.
   */
  async rewrite(request: RewriteRequest, options: RewriteOptions = {}): Promise<RewriteResult> {
    const budget = this.chunkBudget(request.manifest, request.adapterId);
    const tokens = this.measureTokens(request.manifest, request.text);
    if (tokens > budget) return this.rewriteChunked(request, options, this.planSegments(request.manifest, request.text, budget));
    try {
      return await this.rewriteOnce(request, options);
    } catch (error) {
      if (!(error instanceof EchoError) || error.code !== 'OUTPUT_TOO_LONG') throw error;
      const halves = this.splitAfterOverflow(request.manifest, request.text, tokens, error);
      options.onRestart?.();
      return this.rewriteChunked(request, options, halves);
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
   * The most raw-text tokens one call takes: what the frame has room for
   * (more is what `promptIds` refuses with INPUT_TOO_LONG), and no more than
   * a full-ratio rewrite of it fits in `maxNewTokens`. Frame overhead is
   * measured from the manifest itself (an empty prompt build), so it tracks
   * the real frame, not a hardcoded guess.
   */
  private chunkBudget(manifest: ModelManifest, adapterId: StyleId): number {
    const overhead = this.promptIds(manifest, adapterId, '').length;
    const inputLimit = Math.max(1, manifest.limits.maxInputTokens - overhead);
    const outputBudget = Math.max(1, Math.floor(manifest.limits.maxNewTokens / MAX_EXPANSION[adapterId]));
    return Math.min(inputLimit, outputBudget);
  }

  /**
   * `text` as calls of at most `budget` tokens and the separators between
   * them. Pieces are packed into as few calls as fit, cut near equal size so
   * the last call is not a stray sentence rewritten without its context; a
   * piece too big on its own is cut at the next finer split.
   */
  private planSegments(manifest: ModelManifest, text: string, budget: number, level = 0): Segment[] {
    const total = this.measureTokens(manifest, text);
    if (total <= budget) return text ? [{ kind: 'chunk', text }] : [];
    const split = SPLITS[level];
    if (!split) throw new EchoError('INPUT_TOO_LONG', { actual: total, limit: budget });
    const target = Math.ceil(total / Math.ceil(total / budget));
    const parts = text.split(split); // pieces at even indexes, the separators between them at odd ones
    const segments: Segment[] = [];
    let current = '';
    let gap = '';
    const close = () => {
      if (current) segments.push({ kind: 'chunk', text: current });
      current = '';
    };
    for (let index = 0; index < parts.length; index += 2) {
      const piece = parts[index]!;
      if (piece) {
        if (current && (this.measureTokens(manifest, current) >= target || this.measureTokens(manifest, current + gap + piece) > budget)) close();
        if (current) {
          current += gap + piece;
        } else {
          if (gap) segments.push({ kind: 'join', text: gap });
          if (this.measureTokens(manifest, piece) <= budget) current = piece;
          else segments.push(...this.planSegments(manifest, piece, budget, level + 1));
        }
        gap = '';
      }
      gap += parts[index + 1] ?? '';
    }
    close();
    if (gap) segments.push({ kind: 'join', text: gap });
    return segments;
  }

  /** A call that ran past maxNewTokens, as calls of half its size; `overflow` is rethrown when it cannot be cut. */
  private splitAfterOverflow(manifest: ModelManifest, text: string, tokens: number, overflow: EchoError): Segment[] {
    if (tokens < 2) throw overflow;
    let segments: Segment[];
    try {
      // Any piece smaller than the whole is allowed; the near-equal cut still puts it close to the middle.
      segments = this.planSegments(manifest, text, tokens - 1);
    } catch {
      throw overflow;
    }
    if (segments.filter((segment) => segment.kind === 'chunk').length < 2) throw overflow;
    return segments.map((segment) => (segment.kind === 'chunk' ? { ...segment, halved: true } : segment));
  }

  /**
   * Rewrite each chunk in order, streaming the joined text so far through
   * `onPartial`. A chunk whose rewrite runs past maxNewTokens is redone as
   * two halves (once); the chunks before it are kept.
   */
  private async rewriteChunked(request: RewriteRequest, options: RewriteOptions, planned: readonly Segment[]): Promise<RewriteResult> {
    const { manifest, adapterId, adapter } = request;
    const segments = [...planned];
    // Build every chunk's prompt before the first model call, so text the prompt builder refuses (reserved
    // markup) fails at once, as it does for a short text, instead of after the earlier chunks were rewritten.
    for (const segment of segments) {
      if (segment.kind === 'chunk') this.promptIds(manifest, adapterId, segment.text);
    }
    let completed = '';
    let inputTokens = 0;
    let outputTokens = 0;
    for (let index = 0; index < segments.length; index++) {
      const segment = segments[index]!;
      if (options.signal?.aborted) throw new EchoError('CANCELLED');
      if (segment.kind === 'join') {
        completed += segment.text;
        continue;
      }
      const prefix = completed;
      let result: RewriteResult;
      try {
        result = await this.rewriteOnce(
          { manifest, adapterId, adapter, text: segment.text },
          {
            signal: options.signal,
            onProgress: options.onProgress,
            onPartial: options.onPartial ? (readText) => options.onPartial!(() => prefix + readText()) : undefined,
          },
        );
      } catch (error) {
        if (!(error instanceof EchoError) || error.code !== 'OUTPUT_TOO_LONG' || segment.halved) throw error;
        segments.splice(index, 1, ...this.splitAfterOverflow(manifest, segment.text, this.measureTokens(manifest, segment.text), error));
        index--;
        continue;
      }
      completed = prefix + result.text;
      inputTokens += result.inputTokens;
      outputTokens += result.outputTokens;
    }
    const output = completed.trim();
    if (output.length === 0) throw new EchoError('EMPTY_RESULT');
    return { text: output, inputTokens, outputTokens };
  }
}
