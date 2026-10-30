import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { catalogOf, parseAdapterFile, parseModelIndex, resolveModelIndex, type AdapterManifest, type AdapterReference, type ModelManifest } from '../../engine/manifest';
import type { InstalledModelRecord } from '../../shared/modelInstall';
import type { StyleId } from '../../shared/styles';

export const FIXTURES = `${join(__dirname, '..', 'fixtures')}/`;

export function readFixture(path: string): Buffer {
  return readFileSync(`${FIXTURES}${path}`);
}

export function readJsonFixture<T = unknown>(path: string): T {
  return JSON.parse(readFixture(path).toString('utf8')) as T;
}

/** A fixture model as the extension saves it, adapter.json files resolved. */
export function tinyManifest(fixture: 'tiny-echo' | 'tiny-echo-int8'): ModelManifest {
  const index = parseModelIndex(readJsonFixture(`${fixture}/model.json`));
  const adapters: Partial<Record<StyleId, AdapterManifest>> = {};
  for (const [style, reference] of Object.entries(index.adapters) as [StyleId, AdapterReference][]) {
    adapters[style] = parseAdapterFile(readJsonFixture(`${fixture}/${reference.source}${reference.manifest.path}`), style, reference, index);
  }
  return resolveModelIndex(index, adapters);
}

export interface TokenizerProbe {
  readonly text: string;
  readonly ids: number[];
  readonly decoded: string;
}

/** One greedy generation recorded with native onnxruntime, the style's adapter active. */
export interface GenerationCase {
  readonly style: string;
  readonly input: string;
  readonly ids: number[];
  readonly maxNewTokens: number;
  /** EOS included when it was produced. */
  readonly outputIds: number[];
  readonly outputText: string;
  /** The same prompt without any adapter (recorded for the tiny fixtures only). */
  readonly baseOutputIds?: number[];
}

export interface Parity {
  readonly modelId: string;
  readonly tokenizer?: { text: string; ids: number[] }[];
  readonly prompts?: { style: string; input: string; ids: number[] }[];
  readonly generation: GenerationCase[];
}

/** What the tiny fixture's model.json offers: a professional and a grammar adapter, professional as the fallback. */
export const TINY_CATALOG = catalogOf(tinyManifest('tiny-echo'));

export function installedRecord(sourceUrl: string, adapters: StyleId[] = ['professional']): InstalledModelRecord {
  return { sourceUrl, catalog: TINY_CATALOG, adapters };
}
