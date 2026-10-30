import { existsSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { describe, expect, jest, test } from '@jest/globals';
import {
  baseFingerprint,
  catalogOfIndex,
  parseAdapterFile,
  parseModelIndex,
  parseModelManifest,
  type AdapterReference,
} from '../../../engine/manifest';
import { fetchCatalog } from '../../../shared/fetchCatalog';
import type { EngineEvent } from '../../../shared/messages';
import { fetchManifest, sha256Hex } from '../../../worker/modelDownloader';
import { ModelLibrary } from '../../../worker/modelLibrary';
import { ModelStore } from '../../../worker/modelStore';
import { tinyManifest } from '../../helpers/fixtures';
import { MemoryDirectory } from '../../helpers/memoryFileSystem';
import { FIXTURE_URL, indexedModelFiles, modelServer } from '../../helpers/modelServer';

const resolved = tinyManifest('tiny-echo-int8');
const COMMIT = 'a'.repeat(40);
const HUB = (name: string) => `https://huggingface.co/KeelCore/${name}/resolve/${COMMIT}/onnx/`;
const decode = (bytes: Uint8Array) => JSON.parse(new TextDecoder().decode(bytes)) as Record<string, any>;
const encode = (value: unknown) => new TextEncoder().encode(JSON.stringify(value));
const signal = () => new AbortController().signal;

/** Serves the model folder, and each hub folder as that adapter's folder, recording what was asked for. */
function hubFetch(base: (url: string, init?: RequestInit) => Promise<Response>) {
  return (url: string, init?: RequestInit) => {
    const hub = /^https:\/\/huggingface\.co\/KeelCore\/([^/]+)\/resolve\/[0-9a-f]{40}\/onnx\/(.+)$/.exec(url);
    return base(hub ? `${FIXTURE_URL}adapters/${hub[1]}/${hub[2]}` : url, init);
  };
}

function setup(files = indexedModelFiles(), root = new MemoryDirectory()) {
  const events: EngineEvent[] = [];
  const server = modelServer(files);
  const fetch = hubFetch(server.fetch);
  const transport = { fetch, estimate: async () => ({}), digest: sha256Hex, now: () => 0 };
  const library = new ModelLibrary(FIXTURE_URL, {
    openStore: () => ModelStore.open(root),
    transport,
    engine: { unload: jest.fn(async () => undefined), warmUp: jest.fn() },
    emit: (event) => void events.push(event),
  });
  const downloads = () => events.flatMap((event) => (event.type === 'download' ? [event.download] : []));
  return { root, library, server, transport, events, downloads };
}

const index = () => parseModelIndex(decode(indexedModelFiles().get('model.json')!));
const adapterJson = (style = 'professional') => decode(indexedModelFiles().get(`adapters/${style}/adapter.json`)!);
const reference = (style: 'professional' | 'grammar' = 'professional'): AdapterReference => index().adapters[style]!;

describe('a published model.json', () => {
  test('is read as a catalog without fetching any adapter.json', () => {
    const raw = decode(indexedModelFiles().get('model.json')!);
    const catalog = catalogOfIndex(parseModelIndex(raw));
    expect(catalog.adapters.map((adapter) => adapter.style)).toEqual(['professional', 'grammar']);
    expect(catalog.base).toBe(baseFingerprint(resolved));
    expect(catalog.fallbackAdapter).toBe('professional');
  });

  test.each([
    ['another host', `https://evil.example/KeelCore/x/resolve/${COMMIT}/`],
    ['another organization', `https://huggingface.co/Other/x/resolve/${COMMIT}/`],
    ['a branch instead of a commit', 'https://huggingface.co/KeelCore/x/resolve/main/'],
    ['plain http', `http://huggingface.co/KeelCore/x/resolve/${COMMIT}/`],
    ['escaping the commit', `https://huggingface.co/KeelCore/x/resolve/${COMMIT}/../../`],
    ['escaping the model folder', '../adapters/'],
    ['an absolute path', '/adapters/'],
    ['a protocol-relative trick', '\\\\evil.example/'],
    ['a file rather than a folder', 'adapters/professional'],
  ])('refuses an adapter source at %s', (_label, source) => {
    const raw = decode(indexedModelFiles().get('model.json')!);
    raw.adapters.professional.source = source;
    expect(() => parseModelIndex(raw)).toThrow(/adapters\.professional\.source/);
  });

  test('accepts a source pinned to one commit of one of our repositories', () => {
    const raw = decode(indexedModelFiles(() => HUB('KeelCore-750M-Professional')).get('model.json')!);
    expect(parseModelIndex(raw).adapters.professional!.source).toBe(HUB('KeelCore-750M-Professional'));
  });

  test('still needs an adapter, and a fallback that is one of them', () => {
    const raw = decode(indexedModelFiles().get('model.json')!);
    expect(() => parseModelIndex({ ...raw, adapters: {} })).toThrow(/at least one adapter/);
    expect(() => parseModelIndex({ ...raw, fallbackAdapter: 'concise' })).toThrow(/fallbackAdapter concise/);
  });
});

describe('parseAdapterFile', () => {
  test('places files under a relative source in the model folder, and under a hub source by URL', () => {
    const local = parseAdapterFile(adapterJson(), 'professional', reference(), resolved);
    expect(local).toEqual(resolved.adapters.professional);
    const hub = parseAdapterFile(adapterJson(), 'professional', { ...reference(), source: HUB('r') }, resolved);
    expect(hub.files.encoder.source).toBe(HUB('r'));
    expect(hub.files.encoder.path).toBe('encoder.onnx_adapter');
  });

  test.each([
    ['a newer format', { schemaVersion: 2 }, /schemaVersion 2/],
    ['another style', { style: 'grammar' }, /it is for grammar/],
    ['another base', { baseFingerprint: 'x' }, /another base model/],
    ['a file path escaping its folder', { files: { ...adapterJson().files, encoder: { ...adapterJson().files.encoder, path: '../x' } } }, /relative path/],
    ['a size that disagrees with model.json', { files: { ...adapterJson().files, encoder: { ...adapterJson().files.encoder, bytes: 1 } } }, /add up to/],
  ])('refuses %s', (_label, change, message) => {
    expect(() => parseAdapterFile({ ...adapterJson(), ...change }, 'professional', reference(), resolved)).toThrow(message);
  });
});

describe('fetchManifest with adapter references', () => {
  test('resolves every adapter into the manifest the extension saves', async () => {
    const { transport } = setup();
    expect(await fetchManifest(FIXTURE_URL, transport, signal())).toEqual(resolved);
  });

  test('a required adapter whose adapter.json was tampered with fails; the hash catches it', async () => {
    const files = indexedModelFiles();
    const json = decode(files.get('adapters/professional/adapter.json')!);
    files.set('adapters/professional/adapter.json', encode({ ...json, styleToken: '<style:other>' }));
    const { transport } = setup(files);
    await expect(fetchManifest(FIXTURE_URL, transport, signal(), { required: ['professional'] })).rejects.toThrow(/professional adapter/);
  });

  test('an adapter that was not asked for and cannot be read costs only itself', async () => {
    const files = indexedModelFiles();
    files.delete('adapters/grammar/adapter.json');
    const { transport } = setup(files);
    const manifest = await fetchManifest(FIXTURE_URL, transport, signal(), { required: ['professional'] });
    expect(Object.keys(manifest.adapters)).toEqual(['professional']);
  });

  test('a fallback adapter that cannot be read is dropped, so the record stays valid', async () => {
    const files = indexedModelFiles();
    files.delete('adapters/professional/adapter.json');
    const { transport } = setup(files);
    const manifest = await fetchManifest(FIXTURE_URL, transport, signal(), { required: ['grammar'] });
    expect(manifest.fallbackAdapter).toBeNull();
    expect(() => parseModelManifest(JSON.parse(JSON.stringify(manifest)))).not.toThrow();
  });

  test('when no adapter can be read it fails rather than record a model with none', async () => {
    const files = indexedModelFiles();
    files.delete('adapters/professional/adapter.json');
    files.delete('adapters/grammar/adapter.json');
    const { transport } = setup(files);
    await expect(fetchManifest(FIXTURE_URL, transport, signal())).rejects.toMatchObject({ code: 'DOWNLOAD_FAILED' });
  });
});

describe('ModelLibrary with adapters in their own repositories', () => {
  const hubFiles = () => indexedModelFiles((style) => HUB(style));

  test('downloads the base from the model folder and the adapter from its repository, and it survives a restart', async () => {
    const files = hubFiles();
    const root = new MemoryDirectory();
    const first = setup(files, root);
    await first.library.startDownload(['professional']);
    expect(first.downloads().at(-1)).toEqual({ state: 'idle' });
    const { model, store } = await first.library.require();
    expect(model.adapters).toEqual(['professional']);
    expect(model.manifest.adapters.professional!.files.encoder.source).toBe(HUB('professional'));
    expect(model.manifest.adapters.professional!.styleToken).toBe(resolved.adapters.professional!.styleToken);
    expect(await store.read(model.manifest.adapters.professional!.files.encoder)).toEqual(files.get('adapters/professional/encoder.onnx_adapter'));

    // a new worker reads the saved record back
    const again = setup(files, root);
    await again.library.inspect();
    expect((await again.library.require()).model).toEqual(model);
  });

  test('an installed adapter stays usable while its repository cannot be reached', async () => {
    const files = hubFiles();
    const root = new MemoryDirectory();
    const first = setup(files, root);
    await first.library.startDownload(['professional']);

    const offline = new Map(files);
    offline.delete('adapters/professional/adapter.json');
    const second = setup(offline, root);
    await second.library.startDownload(['grammar']);
    expect(second.downloads().at(-1)).toEqual({ state: 'idle' });
    expect((await second.library.require()).model.adapters).toEqual(['professional', 'grammar']);
  });

  test('a corrupted adapter file is caught by the hash in adapter.json and nothing is recorded', async () => {
    const files = hubFiles();
    const bad = new Uint8Array(files.get('adapters/professional/encoder.onnx_adapter')!);
    bad[0] = bad[0]! ^ 1;
    files.set('adapters/professional/encoder.onnx_adapter', bad);
    const { library, downloads } = setup(files);
    await library.startDownload(['professional']);
    expect(downloads().at(-1)).toMatchObject({ state: 'failed' });
    await expect(library.require()).rejects.toMatchObject({ code: 'MODEL_NOT_DOWNLOADED' });
  });
});

describe('the manifests kept in this repository', () => {
  const folder = new URL('../../../../models/keelcore-int8/', import.meta.url);
  const present = existsSync(new URL('adapters/professional/adapter.json', folder));
  (present ? test : test.skip)('model.json and adapter.json agree', () => {
    const modelRaw = JSON.parse(readFileSync(new URL('model.json', folder), 'utf8')) as unknown;
    const model = parseModelIndex(modelRaw);
    const professional = model.adapters.professional!;
    const bytes = readFileSync(new URL('adapters/professional/adapter.json', folder));
    expect(bytes.byteLength).toBe(professional.manifest.bytes);
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(professional.manifest.sha256);
    const adapter = parseAdapterFile(JSON.parse(bytes.toString('utf8')), 'professional', professional, model);
    expect(adapter.styleToken).toBe('<style:professional>');
    expect(adapter.files.encoder.path).toBe('adapters/professional/encoder.onnx_adapter');
  });
});

describe('fetchCatalog', () => {
  test('reads what a folder publishes, without fetching any adapter.json', async () => {
    const fetchImpl = jest.fn(async () => new Response(indexedModelFiles().get('model.json')! as BodyInit));
    expect((await fetchCatalog(fetchImpl as unknown as typeof fetch)).adapters.map((adapter) => adapter.style)).toEqual(['professional', 'grammar']);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  test('refuses an older model.json', async () => {
    const raw = decode(indexedModelFiles().get('model.json')!);
    const serve = (async () => new Response(encode({ ...raw, schemaVersion: 2 }) as BodyInit)) as unknown as typeof fetch;
    await expect(fetchCatalog(serve)).rejects.toThrow(/schemaVersion 2 is not supported/);
  });
});
