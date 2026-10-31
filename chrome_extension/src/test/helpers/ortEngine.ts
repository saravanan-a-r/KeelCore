/**
 * The base model and its adapters from model files, running onnxruntime-web
 * under Node -- the same runtime the extension's worker uses.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-web';
import { parseAdapterFile, parseModelIndex, resolveModelIndex, type AdapterManifest, type AdapterReference, type ModelManifest } from '../../engine/manifest';
import type { TensorFactory } from '../../engine/onnx/crossKvRunner';
import { RewriteEngine } from '../../engine/rewriteEngine';
import type { StyleId } from '../../shared/styles';
import type { LoadedAdapter } from '../../worker/modelLoader';

ort.env.wasm.numThreads = 1;

type ReadFile = (path: string) => Uint8Array | Promise<Uint8Array>;

export interface TestModel {
  readonly manifest: ModelManifest;
  readonly engine: RewriteEngine;
  /** `style`'s adapter, loaded; the caller releases it. */
  adapter(style: StyleId): Promise<LoadedAdapter>;
}

async function modelFromFiles(manifest: ModelManifest, read: ReadFile): Promise<TestModel> {
  const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all', logSeverityLevel: 3 } as const;
  const engine = new RewriteEngine({
    geometry: manifest.architecture,
    tokenizerJson: JSON.parse(new TextDecoder().decode(await read(manifest.files.tokenizer.path))),
    encoder: await ort.InferenceSession.create(await read(manifest.files.encoder.path), options),
    decoder: await ort.InferenceSession.create(await read(manifest.files.decoder.path), options),
    TensorCtor: ort.Tensor as unknown as TensorFactory,
  });
  return {
    manifest,
    engine,
    async adapter(style) {
      const files = manifest.adapters[style]!.files;
      const encoder = await ort.LoraAdapter.create(await read(files.encoder.path));
      const decoder = await ort.LoraAdapter.create(await read(files.decoder.path));
      return { encoder, decoder, release: async () => void (await Promise.all([encoder.release(), decoder.release()])) };
    },
  };
}

/** The model.json in a folder, with its adapter.json files resolved. */
async function readManifest(read: (path: string) => Uint8Array | Promise<Uint8Array>): Promise<ModelManifest> {
  const json = async (path: string): Promise<unknown> => JSON.parse(new TextDecoder().decode(await read(path)));
  const index = parseModelIndex(await json('model.json'));
  const adapters: Partial<Record<StyleId, AdapterManifest>> = {};
  for (const [style, reference] of Object.entries(index.adapters) as [StyleId, AdapterReference][]) {
    adapters[style] = parseAdapterFile(await json(`${reference.source}${reference.manifest.path}`), style, reference, index);
  }
  return resolveModelIndex(index, adapters);
}

export async function modelFromFolder(folder: string): Promise<TestModel> {
  const manifest = await readManifest((path) => new Uint8Array(readFileSync(join(folder, path))));
  return modelFromFiles(manifest, (path) => new Uint8Array(readFileSync(join(folder, path))));
}
