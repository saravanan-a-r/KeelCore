/**
 * The Chrome Web Store package: dist/ as one zip, rebuilt by every production
 * build. It has the same files as dist/ except what the Store should never
 * get: macOS Finder droppings (.DS_Store), any other hidden file, source maps
 * and Thumbs.db. Each build removes the zips of earlier builds, so release/
 * holds exactly one, the one to upload.
 *
 * Written here, not with a zip library or the system `zip`: it needs nothing
 * but Node and gives the same bytes on every machine for the same dist/
 * (sorted names, fixed timestamps).
 */
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { deflateRawSync } from 'node:zlib';
import type { Compiler } from 'webpack';

const PLUGIN = 'EchoMeBetterZipPackage';

/** Files that stay out of the package, by base name or extension. */
const EXCLUDED = /(^\.|\.map$|^Thumbs\.db$)/;

/** 1980-01-01 00:00, the zip format's earliest date, so the same files always give the same zip. */
const DOS_DATE = (0 << 9) | (1 << 5) | 1;
const DOS_TIME = 0;
const UTF8_NAMES = 1 << 11;

const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let bit = 0; bit < 8; bit += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of data) crc = CRC_TABLE[(crc ^ byte) & 0xff]! ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

/** Every file under `root`, as `/`-separated paths relative to it, sorted, without the excluded ones. */
export function packageFiles(root: string): string[] {
  const found: string[] = [];
  const walk = (folder: string): void => {
    for (const entry of readdirSync(folder, { withFileTypes: true })) {
      if (EXCLUDED.test(entry.name)) continue;
      const path = join(folder, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (entry.isFile()) found.push(relative(root, path).split(sep).join('/'));
    }
  };
  walk(root);
  return found.sort();
}

function u16(value: number): Buffer {
  const buffer = Buffer.alloc(2);
  buffer.writeUInt16LE(value);
  return buffer;
}

function u32(value: number): Buffer {
  const buffer = Buffer.alloc(4);
  buffer.writeUInt32LE(value);
  return buffer;
}

/** The zip file's bytes for the files under `root` (see packageFiles). */
export function zipFolder(root: string): Buffer {
  const parts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  const names = packageFiles(root);
  for (const name of names) {
    const data = readFileSync(join(root, name));
    const packed = deflateRawSync(data, { level: 9 });
    const deflated = packed.length < data.length;
    const body = deflated ? packed : data;
    const method = deflated ? 8 : 0;
    const nameBytes = Buffer.from(name, 'utf8');
    const crc = crc32(data);
    const common = [u16(20), u16(UTF8_NAMES), u16(method), u16(DOS_TIME), u16(DOS_DATE), u32(crc), u32(body.length), u32(data.length), u16(nameBytes.length), u16(0)];
    const local = Buffer.concat([u32(0x04034b50), ...common, nameBytes, body]);
    directory.push(Buffer.concat([u32(0x02014b50), u16(20), ...common, u16(0), u16(0), u16(0), u32(0), u32(offset), nameBytes]));
    parts.push(local);
    offset += local.length;
  }
  const central = Buffer.concat(directory);
  const end = Buffer.concat([u32(0x06054b50), u16(0), u16(0), u16(names.length), u16(names.length), u32(central.length), u32(offset), u16(0)]);
  return Buffer.concat([...parts, central, end]);
}

/**
 * Webpack plugin: after a production build has written dist/, zip it into
 * `releaseDir`/`name`-`version`.zip and remove the zips left by earlier builds.
 * Put it on the build that finishes last (the foreground one depends on the
 * extension one), so dist/ is complete.
 */
export class ZipPackagePlugin {
  constructor(private readonly options: { distDir: string; releaseDir: string; name: string; version: string }) {}

  apply(compiler: Compiler): void {
    compiler.hooks.done.tap(PLUGIN, (stats) => {
      if (stats.hasErrors()) return;
      const { distDir, releaseDir, name, version } = this.options;
      mkdirSync(releaseDir, { recursive: true });
      for (const old of readdirSync(releaseDir)) if (old.endsWith('.zip')) rmSync(join(releaseDir, old));
      const file = join(releaseDir, `${name}-${version}.zip`);
      const zip = zipFolder(distDir);
      writeFileSync(file, zip);
      console.log(`\nChrome Web Store package: ${relative(process.cwd(), file)} (${(zip.length / 1024 / 1024).toFixed(1)} MB, ${packageFiles(distDir).length} files)`);
    });
  }
}
