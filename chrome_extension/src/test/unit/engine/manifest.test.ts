import { describe, expect, test } from '@jest/globals';
import { adapterFiles, baseFiles, bytesOf, catalogOf, parseModelManifest, relabelSavedManifest } from '../../../engine/manifest';
import { readJsonFixture } from '../../helpers/fixtures';

const valid = readJsonFixture<Record<string, unknown>>('tiny-echo/model.json');
const without = (key: string) => Object.fromEntries(Object.entries(valid).filter(([k]) => k !== key));
const adapters = valid.adapters as Record<string, unknown>;

describe('parseModelManifest', () => {
  test('accepts a manifest written by the packaging tool, and reads back what it produced', () => {
    const manifest = parseModelManifest(valid);
    expect(manifest.ioContract).toBe('cross-kv/1');
    expect(Object.keys(manifest.adapters)).toEqual(['professional', 'grammar']);
    expect(manifest.fallbackAdapter).toBe('professional');
    // The store saves the parsed manifest and parses it again on every read.
    expect(parseModelManifest(JSON.parse(JSON.stringify(manifest)))).toEqual(manifest);
  });

  test('the catalog sizes the base and each adapter separately', () => {
    const manifest = parseModelManifest(valid);
    const catalog = catalogOf(manifest);
    expect(catalog.model.sizeBytes).toBe(bytesOf(baseFiles(manifest)));
    expect(catalog.adapters).toEqual([
      { style: 'professional', sizeBytes: bytesOf(adapterFiles(manifest.adapters.professional!)) },
      { style: 'grammar', sizeBytes: bytesOf(adapterFiles(manifest.adapters.grammar!)) },
    ]);
  });

  test('the exempt token list defaults to the plain ban and is read when present', () => {
    const plain = parseModelManifest(valid).generation;
    expect(plain.noRepeatExemptTokenIds).toEqual([]);
    const generation = { ...(valid.generation as object), noRepeatExemptTokenIds: [3, 4] };
    const manifest = parseModelManifest({ ...valid, generation });
    expect(manifest.generation).toMatchObject({ noRepeatExemptTokenIds: [3, 4] });
    expect(parseModelManifest(JSON.parse(JSON.stringify(manifest)))).toEqual(manifest);
  });

  test.each([
    ['an exempt id outside the vocabulary', { noRepeatExemptTokenIds: [400] }, /noRepeatExemptTokenIds/],
    ['a negative exempt id', { noRepeatExemptTokenIds: [-1] }, /noRepeatExemptTokenIds/],
    ['exempt ids that are not a list', { noRepeatExemptTokenIds: 3 }, /noRepeatExemptTokenIds/],
  ])('rejects %s', (_label, extra, message) => {
    expect(() => parseModelManifest({ ...valid, generation: { ...(valid.generation as object), ...extra } })).toThrow(message);
  });

  test('a manifest whose adapters cover every style needs no fallback', () => {
    expect(parseModelManifest(without('fallbackAdapter')).fallbackAdapter).toBeNull();
  });

  test.each([
    ['an older schema version', { ...valid, schemaVersion: 1 }, /schemaVersion 1 is not supported/],
    ['a different I/O contract', { ...valid, ioContract: 'other/1' }, /ioContract other\/1 is not supported/],
    ['a prompt with no input segment', { ...valid, prompt: [{ styleToken: true }, { token: '</s>' }] }, /exactly one \{"input": true\}/],
    ['a prompt with no style token', { ...valid, prompt: [{ input: true }, { token: '</s>' }] }, /exactly one \{"styleToken": true\}/],
    ['an adapter not named after a style', { ...valid, adapters: { ...adapters, casual: adapters.grammar } }, /adapters\.casual/],
    ['no adapters at all', { ...valid, adapters: {} }, /at least one adapter/],
    ['a fallback that is not one of the adapters', { ...valid, fallbackAdapter: 'concise' }, /fallbackAdapter concise is not in adapters/],
    ['a file path escaping the model folder', { ...valid, files: { ...(valid.files as object), encoder: { path: '../x.onnx', bytes: 1, sha256: 'a' } } }, /relative path inside/],
    ['an unknown input transform', { ...valid, inputTransforms: ['rot13'] }, /inputTransforms/],
    ['missing limits', without('limits'), /limits must be an object/],
  ])('rejects %s', (_label, manifest, message) => {
    expect(() => parseModelManifest(manifest)).toThrow(message);
  });
});

describe('relabelSavedManifest', () => {
  test('gives a saved manifest of the same interface revision the current label, so it parses', () => {
    const saved = { ...valid, ioContract: 'earlier-label/1' };
    expect(() => parseModelManifest(saved)).toThrow(/ioContract earlier-label\/1 is not supported/);
    expect(parseModelManifest(relabelSavedManifest(saved)).ioContract).toBe('cross-kv/1');
  });

  test('leaves another revision alone, so it is still rejected', () => {
    const saved = { ...valid, ioContract: 'cross-kv/2' };
    expect(relabelSavedManifest(saved)).toBe(saved);
    expect(() => parseModelManifest(relabelSavedManifest(saved))).toThrow(/ioContract cross-kv\/2 is not supported/);
  });

  test.each([[null], ['text'], [[1]], [{ ...valid, ioContract: undefined }]])('passes %j through unchanged', (raw) => {
    expect(relabelSavedManifest(raw)).toBe(raw);
  });
});
