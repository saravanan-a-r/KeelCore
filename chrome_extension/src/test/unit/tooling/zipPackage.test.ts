import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { inflateRawSync } from 'node:zlib';
import { afterEach, beforeEach, describe, expect, test } from '@jest/globals';
import { packageFiles, zipFolder } from '../../../../tooling/zipPackage';

/** What the zip holds, read back from its central directory, the way an unzipper does. */
function readZip(zip: Buffer): Map<string, Buffer> {
  const end = zip.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
  const count = zip.readUInt16LE(end + 10);
  let at = zip.readUInt32LE(end + 16);
  const files = new Map<string, Buffer>();
  for (let i = 0; i < count; i += 1) {
    expect(zip.readUInt32LE(at)).toBe(0x02014b50);
    const method = zip.readUInt16LE(at + 10);
    const crc = zip.readUInt32LE(at + 16);
    const packedSize = zip.readUInt32LE(at + 20);
    const size = zip.readUInt32LE(at + 24);
    const nameLength = zip.readUInt16LE(at + 28);
    const offset = zip.readUInt32LE(at + 42);
    const name = zip.subarray(at + 46, at + 46 + nameLength).toString('utf8');
    const start = offset + 30 + zip.readUInt16LE(offset + 26) + zip.readUInt16LE(offset + 28);
    const body = zip.subarray(start, start + packedSize);
    const data = method === 8 ? inflateRawSync(body) : body;
    expect(data.length).toBe(size);
    expect(crcOf(data)).toBe(crc);
    files.set(name, Buffer.from(data));
    at += 46 + nameLength + zip.readUInt16LE(at + 30) + zip.readUInt16LE(at + 32);
  }
  return files;
}

function crcOf(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of data) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) crc = crc & 1 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

describe('the Chrome Web Store zip', () => {
  let root: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'echomebetter-zip-'));
    mkdirSync(join(root, 'ui', 'popup'), { recursive: true });
    writeFileSync(join(root, 'manifest.json'), '{"name":"EchoMeBetter"}\n');
    writeFileSync(join(root, 'ui', 'popup', 'popup.html'), '<!doctype html>'.repeat(200));
    writeFileSync(join(root, 'ui', 'ünï.txt'), 'accents in a name');
    writeFileSync(join(root, 'empty.bin'), '');
    writeFileSync(join(root, '.DS_Store'), 'finder');
    writeFileSync(join(root, 'ui', '.DS_Store'), 'finder');
    writeFileSync(join(root, 'background.js.map'), '{}');
    writeFileSync(join(root, 'Thumbs.db'), 'windows');
  });
  afterEach(() => rmSync(root, { recursive: true, force: true }));

  test('leaves out .DS_Store and other hidden, map and Thumbs.db files, at any depth', () => {
    expect(packageFiles(root)).toEqual(['empty.bin', 'manifest.json', 'ui/popup/popup.html', 'ui/ünï.txt']);
  });

  test('holds every other file, byte for byte, under its path relative to the folder', () => {
    const files = readZip(zipFolder(root));
    expect([...files.keys()]).toEqual(['empty.bin', 'manifest.json', 'ui/popup/popup.html', 'ui/ünï.txt']);
    expect(files.get('manifest.json')?.toString()).toBe('{"name":"EchoMeBetter"}\n');
    expect(files.get('ui/popup/popup.html')?.toString()).toBe('<!doctype html>'.repeat(200));
    expect(files.get('ui/ünï.txt')?.toString()).toBe('accents in a name');
    expect(files.get('empty.bin')?.length).toBe(0);
  });

  test('gives the same bytes for the same files', () => {
    expect(zipFolder(root).equals(zipFolder(root))).toBe(true);
  });
});
