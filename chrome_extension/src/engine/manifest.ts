/**
 * `model.json`: everything model-specific, as data.
 *
 * KeelCore is a base model plus one LoRA adapter per writing style. The
 * extension's code knows how to run an encoder-decoder that honours
 * the model's cached-attention I/O contract (`ioContract`, see the runner in
 * onnx/) with an adapter
 * active. Which files make up the base and each style's adapter, the prompt
 * frame, the special tokens and the length limits all come from this file,
 * served next to the model files.
 *
 * Adapters are keyed by style. A style without an adapter of its own (still
 * in training) runs with `fallbackAdapter`, under that adapter's style token;
 * a newly trained adapter is one more entry under `adapters`, no code changes.
 *
 * A model folder publishes `parseModelIndex`'s form: each adapter is a reference
 * to its own folder, which holds an `adapter.json` naming its files. Once those
 * are resolved the extension saves `parseModelManifest`'s form, with the adapter
 * files inline, so an installed model needs no network.
 *
 * The manifest is validated in full before anything is downloaded or loaded,
 * so a mismatched or hand-edited model fails with a precise message instead
 * of producing wrong text later.
 */
import type { AdapterSummary, ModelCatalog } from '../shared/modelInstall';
import type { ModelSummary } from '../shared/status';
import { isStyleId, type StyleId } from '../shared/styles';

export const SUPPORTED_SCHEMA_VERSION = 3;
export const ADAPTER_FILE_SCHEMA_VERSION = 1;
export const SUPPORTED_IO_CONTRACT = 'cross-kv/1';

export const TEXT_TRANSFORMS = ['escape-spiece-markers'] as const;

export type TextTransformName = (typeof TEXT_TRANSFORMS)[number];

/**
 * The encoder input, one segment at a time: a vocabulary piece by name (frame
 * markers, `</s>`), the active adapter's style token, or the user's text.
 */
export type PromptSegment = { readonly token: string } | { readonly styleToken: true } | { readonly input: true };

export interface ModelFile {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
  /** The folder `path` is relative to, when it is not the model's own. */
  readonly source?: string;
}

/** A model.json's entry for one adapter: where it is, and the hash that vouches for its adapter.json. */
export interface AdapterReference {
  /** A folder: relative to the model's, or pinned to one commit of a repository of ours. */
  readonly source: string;
  /** The adapter.json, with `path` relative to `source`. */
  readonly manifest: ModelFile;
  /** The size of the adapter's files, for showing before anything is fetched. */
  readonly sizeBytes: number;
}

export interface AdapterManifest {
  /** The style marker the adapter was trained with. */
  readonly styleToken: string;
  /** One half per graph: each session rejects weights it does not have. */
  readonly files: { readonly encoder: ModelFile; readonly decoder: ModelFile };
}

export interface ModelManifest {
  readonly schemaVersion: number;
  readonly ioContract: string;
  readonly id: string;
  readonly displayName: string;
  readonly precision: string;
  readonly files: { readonly encoder: ModelFile; readonly decoder: ModelFile; readonly tokenizer: ModelFile };
  readonly architecture: {
    readonly numDecoderLayers: number;
    readonly numHeads: number;
    readonly headDim: number;
    readonly vocabSize: number;
  };
  readonly tokens: { readonly decoderStartId: number; readonly eosId: number; readonly padId: number };
  readonly limits: { readonly maxInputTokens: number; readonly maxNewTokens: number };
  readonly generation: {
    readonly repetitionPenalty: number;
    readonly noRepeatNgramSize: number;
    /** Token IDs the n-gram ban never applies to (the digits). */
    readonly noRepeatExemptTokenIds: readonly number[];
  };
  readonly inputTransforms: readonly TextTransformName[];
  readonly reservedInputTokens: readonly string[];
  readonly prompt: readonly PromptSegment[];
  readonly adapters: Readonly<Partial<Record<StyleId, AdapterManifest>>>;
  /** The adapter a style without one of its own runs with; null once every style has its own. */
  readonly fallbackAdapter: StyleId | null;
}

/** A model.json as published: the base in full, the adapters by reference. */
export interface ModelIndex extends Omit<ModelManifest, 'adapters'> {
  readonly adapters: Readonly<Partial<Record<StyleId, AdapterReference>>>;
}

export class ManifestError extends Error {
  constructor(message: string) {
    super(`model.json: ${message}`);
    this.name = 'ManifestError';
  }
}

type Json = Record<string, unknown>;

function object(value: unknown, where: string): Json {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new ManifestError(`${where} must be an object`);
  return value as Json;
}

function string(value: unknown, where: string): string {
  if (typeof value !== 'string' || value.length === 0) throw new ManifestError(`${where} must be a non-empty string`);
  return value;
}

function integer(value: unknown, where: string, min: number): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min) {
    throw new ManifestError(`${where} must be an integer >= ${min}`);
  }
  return value;
}

function file(value: unknown, where: string): ModelFile {
  const json = object(value, where);
  const path = string(json.path, `${where}.path`);
  if (!isRelativePath(path)) throw new ManifestError(`${where}.path must be a relative path inside the model folder`);
  // The hash is also the file's name in the extension's model storage.
  const sha256 = string(json.sha256, `${where}.sha256`);
  if (!/^[0-9a-f]{64}$/.test(sha256)) throw new ManifestError(`${where}.sha256 must be 64 lowercase hex characters`);
  const parsed = { path, bytes: integer(json.bytes, `${where}.bytes`, 1), sha256 };
  if (json.source === undefined) return parsed;
  if (typeof json.source !== 'string' || !isPinnedSource(json.source)) throw new ManifestError(`${where}.source must be a folder at one commit of a KeelCore repository`);
  return { ...parsed, source: json.source };
}

function isRelativePath(path: string): boolean {
  return !path.split('/').includes('..') && !path.startsWith('/') && !path.includes('\\') && !/^[a-z][a-z0-9+.-]*:/i.test(path);
}

/** The only place outside the model's folder files may come from: one commit, so what it names cannot change. */
const PINNED_SOURCE = /^https:\/\/huggingface\.co\/KeelCore\/[A-Za-z0-9][A-Za-z0-9._-]*\/resolve\/[0-9a-f]{40}\/(?:[A-Za-z0-9._-]+\/)*$/;

function isPinnedSource(source: string): boolean {
  return PINNED_SOURCE.test(source) && !source.split('/').some((part) => part === '.' || part === '..');
}

function adapterSource(value: unknown, where: string): string {
  const source = string(value, where);
  if (isPinnedSource(source) || (source.endsWith('/') && isRelativePath(source))) return source;
  throw new ManifestError(`${where} must be a folder inside the model folder, or at one commit of a KeelCore repository`);
}

function prompt(value: unknown): PromptSegment[] {
  if (!Array.isArray(value) || value.length === 0) throw new ManifestError('prompt must be a non-empty array');
  const segments = value.map((raw, index): PromptSegment => {
    const segment = object(raw, `prompt[${index}]`);
    if (segment.input === true) return { input: true };
    if (segment.styleToken === true) return { styleToken: true };
    if (typeof segment.token === 'string') return { token: string(segment.token, `prompt[${index}].token`) };
    throw new ManifestError(`prompt[${index}] must be {"token"}, {"styleToken": true} or {"input": true}`);
  });
  for (const kind of ['input', 'styleToken'] as const) {
    if (segments.filter((segment) => kind in segment).length !== 1) {
      throw new ManifestError(`prompt must contain exactly one {"${kind}": true} segment`);
    }
  }
  return segments;
}

function adapters(value: unknown): Partial<Record<StyleId, AdapterManifest>> {
  const json = object(value, 'adapters');
  const parsed: Partial<Record<StyleId, AdapterManifest>> = {};
  for (const [id, raw] of Object.entries(json)) {
    if (!isStyleId(id)) throw new ManifestError(`adapters.${id}: adapters are named after the style they are trained for`);
    const adapter = object(raw, `adapters.${id}`);
    const files = object(adapter.files, `adapters.${id}.files`);
    parsed[id] = {
      styleToken: string(adapter.styleToken, `adapters.${id}.styleToken`),
      files: { encoder: file(files.encoder, `adapters.${id}.files.encoder`), decoder: file(files.decoder, `adapters.${id}.files.decoder`) },
    };
  }
  if (Object.keys(parsed).length === 0) throw new ManifestError('adapters must name at least one adapter');
  return parsed;
}

function adapterReferences(value: unknown): Partial<Record<StyleId, AdapterReference>> {
  const json = object(value, 'adapters');
  const parsed: Partial<Record<StyleId, AdapterReference>> = {};
  for (const [id, raw] of Object.entries(json)) {
    if (!isStyleId(id)) throw new ManifestError(`adapters.${id}: adapters are named after the style they are trained for`);
    const reference = object(raw, `adapters.${id}`);
    parsed[id] = {
      source: adapterSource(reference.source, `adapters.${id}.source`),
      manifest: file(reference.manifest, `adapters.${id}.manifest`),
      sizeBytes: integer(reference.sizeBytes, `adapters.${id}.sizeBytes`, 1),
    };
  }
  if (Object.keys(parsed).length === 0) throw new ManifestError('adapters must name at least one adapter');
  return parsed;
}

export function parseModelIndex(raw: unknown): ModelIndex {
  const json = object(raw, 'manifest');
  if (json.schemaVersion !== SUPPORTED_SCHEMA_VERSION) {
    throw new ManifestError(`schemaVersion ${String(json.schemaVersion)} is not supported (expected ${SUPPORTED_SCHEMA_VERSION})`);
  }
  return { ...parseBase(json), ...parseAdapterChoice(json, adapterReferences(json.adapters)) };
}

/**
 * An adapter.json, checked against the model that refers to it, as the entry a
 * saved manifest has for it. Its file paths are relative to its own folder.
 */
export function parseAdapterFile(raw: unknown, style: StyleId, reference: AdapterReference, index: Pick<ModelManifest, 'files'>): AdapterManifest {
  const json = object(raw, 'adapter.json');
  if (json.schemaVersion !== ADAPTER_FILE_SCHEMA_VERSION) {
    throw new ManifestError(`adapter.json: schemaVersion ${String(json.schemaVersion)} is not supported (expected ${ADAPTER_FILE_SCHEMA_VERSION})`);
  }
  if (json.style !== style) throw new ManifestError(`adapter.json: it is for ${String(json.style)}, not ${style}`);
  if (json.baseFingerprint !== baseFingerprint(index)) throw new ManifestError(`adapter.json: the ${style} adapter was trained on another base model`);
  const files = object(json.files, 'adapter.json files');
  const place = (parsed: ModelFile): ModelFile =>
    isPinnedSource(reference.source) ? { ...parsed, source: reference.source } : { ...parsed, path: `${reference.source}${parsed.path}` };
  const encoder = place(file(files.encoder, 'adapter.json files.encoder'));
  const decoder = place(file(files.decoder, 'adapter.json files.decoder'));
  if (encoder.bytes + decoder.bytes !== reference.sizeBytes) {
    throw new ManifestError(`adapter.json: its files add up to ${encoder.bytes + decoder.bytes} bytes, model.json says ${reference.sizeBytes}`);
  }
  return { styleToken: string(json.styleToken, 'adapter.json styleToken'), files: { encoder, decoder } };
}

/** The manifest to save for `index`, with `adapters` resolved. A fallback that could not be resolved is dropped. */
export function resolveModelIndex(index: ModelIndex, adapters: Partial<Record<StyleId, AdapterManifest>>): ModelManifest {
  const fallbackAdapter = index.fallbackAdapter && index.fallbackAdapter in adapters ? index.fallbackAdapter : null;
  return { ...index, adapters, fallbackAdapter };
}

export function parseModelManifest(raw: unknown): ModelManifest {
  const json = object(raw, 'manifest');
  if (json.schemaVersion !== SUPPORTED_SCHEMA_VERSION) {
    throw new ManifestError(`schemaVersion ${String(json.schemaVersion)} is not supported (expected ${SUPPORTED_SCHEMA_VERSION})`);
  }
  return { ...parseBase(json), ...parseAdapterChoice(json, adapters(json.adapters)) };
}

function parseAdapterChoice<T>(json: Json, parsedAdapters: Partial<Record<StyleId, T>>): { adapters: Partial<Record<StyleId, T>>; fallbackAdapter: StyleId | null } {
  const fallback = json.fallbackAdapter ?? null;
  if (fallback !== null && !(isStyleId(fallback) && fallback in parsedAdapters)) {
    throw new ManifestError(`fallbackAdapter ${String(fallback)} is not in adapters`);
  }
  return { adapters: parsedAdapters, fallbackAdapter: fallback };
}

/** Everything a model.json says about the base, common to both schemas. */
function parseBase(json: Json): Omit<ModelManifest, 'adapters' | 'fallbackAdapter'> {
  if (json.ioContract !== SUPPORTED_IO_CONTRACT) {
    throw new ManifestError(`ioContract ${String(json.ioContract)} is not supported (expected ${SUPPORTED_IO_CONTRACT})`);
  }

  const files = object(json.files, 'files');
  const architecture = object(json.architecture, 'architecture');
  const tokens = object(json.tokens, 'tokens');
  const limits = object(json.limits, 'limits');
  const generation = object(json.generation ?? {}, 'generation');

  const inputTransforms = (json.inputTransforms ?? []) as unknown;
  if (!Array.isArray(inputTransforms) || !inputTransforms.every((name) => (TEXT_TRANSFORMS as readonly unknown[]).includes(name))) {
    throw new ManifestError(`inputTransforms must be a list drawn from ${TEXT_TRANSFORMS.join(', ')}`);
  }
  const reserved = (json.reservedInputTokens ?? []) as unknown;
  if (!Array.isArray(reserved) || !reserved.every((token) => typeof token === 'string')) {
    throw new ManifestError('reservedInputTokens must be a list of strings');
  }

  const repetitionPenalty = generation.repetitionPenalty ?? 1;
  if (typeof repetitionPenalty !== 'number' || !(repetitionPenalty > 0)) {
    throw new ManifestError('generation.repetitionPenalty must be a positive number');
  }

  const exemptIds = generation.noRepeatExemptTokenIds ?? [];
  const vocabSize = integer(architecture.vocabSize, 'architecture.vocabSize', 2);
  if (!Array.isArray(exemptIds) || !exemptIds.every((id) => Number.isInteger(id) && id >= 0 && id < vocabSize)) {
    throw new ManifestError('generation.noRepeatExemptTokenIds must be a list of token IDs inside the vocabulary');
  }

  return {
    schemaVersion: SUPPORTED_SCHEMA_VERSION,
    ioContract: SUPPORTED_IO_CONTRACT,
    id: string(json.id, 'id'),
    displayName: string(json.displayName, 'displayName'),
    precision: typeof json.precision === 'string' ? json.precision : 'unknown',
    files: {
      encoder: file(files.encoder, 'files.encoder'),
      decoder: file(files.decoder, 'files.decoder'),
      tokenizer: file(files.tokenizer, 'files.tokenizer'),
    },
    architecture: {
      numDecoderLayers: integer(architecture.numDecoderLayers, 'architecture.numDecoderLayers', 1),
      numHeads: integer(architecture.numHeads, 'architecture.numHeads', 1),
      headDim: integer(architecture.headDim, 'architecture.headDim', 1),
      vocabSize: integer(architecture.vocabSize, 'architecture.vocabSize', 2),
    },
    tokens: {
      decoderStartId: integer(tokens.decoderStartId, 'tokens.decoderStartId', 0),
      eosId: integer(tokens.eosId, 'tokens.eosId', 0),
      padId: integer(tokens.padId, 'tokens.padId', 0),
    },
    limits: {
      maxInputTokens: integer(limits.maxInputTokens, 'limits.maxInputTokens', 1),
      maxNewTokens: integer(limits.maxNewTokens, 'limits.maxNewTokens', 1),
    },
    generation: {
      repetitionPenalty,
      noRepeatNgramSize: integer(generation.noRepeatNgramSize ?? 0, 'generation.noRepeatNgramSize', 0),
      noRepeatExemptTokenIds: exemptIds as number[],
    },
    inputTransforms: inputTransforms as TextTransformName[],
    reservedInputTokens: reserved as string[],
    prompt: prompt(json.prompt),
  };
}

/** The base model's files, smallest first. */
export function baseFiles(manifest: Pick<ModelManifest, 'files'>): ModelFile[] {
  const { encoder, decoder, tokenizer } = manifest.files;
  return [tokenizer, encoder, decoder];
}

export function adapterFiles(adapter: AdapterManifest): ModelFile[] {
  return [adapter.files.encoder, adapter.files.decoder];
}

export function bytesOf(files: readonly ModelFile[]): number {
  return files.reduce((sum, file) => sum + file.bytes, 0);
}

/**
 * Identifies the base weights. Adapters are trained against one base, so a
 * manifest with a different base cannot add adapters to an installed one.
 */
export function baseFingerprint(manifest: Pick<ModelManifest, 'files'>): string {
  return baseFiles(manifest)
    .map((file) => file.sha256)
    .join(':');
}

export function summarize(manifest: Pick<ModelManifest, 'id' | 'displayName' | 'precision' | 'files'>): ModelSummary {
  return {
    id: manifest.id,
    displayName: manifest.displayName,
    precision: manifest.precision,
    sizeBytes: bytesOf(baseFiles(manifest)),
  };
}

/** What the manifest offers, as the extension's pages show it. */
export function catalogOf(manifest: ModelManifest): ModelCatalog {
  const adapterList: AdapterSummary[] = [];
  for (const [style, adapter] of Object.entries(manifest.adapters) as [StyleId, AdapterManifest][]) {
    adapterList.push({ style, sizeBytes: bytesOf(adapterFiles(adapter)) });
  }
  return { base: baseFingerprint(manifest), model: summarize(manifest), adapters: adapterList, fallbackAdapter: manifest.fallbackAdapter };
}

/** What a published model.json offers, read without fetching any adapter.json. */
export function catalogOfIndex(index: ModelIndex): ModelCatalog {
  const adapterList = (Object.entries(index.adapters) as [StyleId, AdapterReference][]).map(([style, reference]) => ({ style, sizeBytes: reference.sizeBytes }));
  return { base: baseFingerprint(index), model: summarize(index), adapters: adapterList, fallbackAdapter: index.fallbackAdapter };
}
