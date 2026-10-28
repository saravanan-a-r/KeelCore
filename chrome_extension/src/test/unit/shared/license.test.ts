/**
 * The agreement to the EchoMeBetter License: what is kept, and the one place
 * a download is allowed to start (the service worker's gate).
 */
import { describe, expect, jest, test } from '@jest/globals';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { downloadUnderLicense, licenseAccepted } from '../../../background/licenseGate';
import { describeError } from '../../../shared/errors';
import { acceptsCurrentLicense, LICENSE_DATE, LICENSE_STORAGE_KEY, LICENSE_VERSION, loadLicenseAcceptance, parseLicenseAcceptance, recordLicenseAcceptance } from '../../../shared/license';
import { isOpenLicenseRequest, isSetupEvent, isSetupRequest, isUiRequest } from '../../../shared/messages';
import { LICENSE_BANNER } from '../../../../tooling/licenseBanner';

/** chrome.storage.local, in memory. */
function memoryStorage(initial: Record<string, unknown> = {}) {
  const data: Record<string, unknown> = { ...initial };
  const storage = {
    get: jest.fn(async (key: string) => (key in data ? { [key]: data[key] } : {})),
    set: jest.fn(async (items: Record<string, unknown>) => void Object.assign(data, items)),
  } as unknown as chrome.storage.StorageArea;
  return { storage, data };
}

const current = { version: LICENSE_VERSION, date: LICENSE_DATE, acceptedAt: 1_000 };
const ROOT = join(__dirname, '..', '..', '..', '..', '..');

describe('the license files', () => {
  test('LICENSE, NOTICE and the build banner name the version and date the extension asks the user to agree to', () => {
    const license = readFileSync(join(ROOT, 'LICENSE'), 'utf8');
    expect(license).toContain(`Version ${LICENSE_VERSION} (Personal and Research Use)`);
    expect(license).toContain(`Version Date: ${LICENSE_DATE}`);
    const notice = readFileSync(join(ROOT, 'NOTICE'), 'utf8').replace(/\s+/g, ' ');
    expect(notice).toContain(`Version ${LICENSE_VERSION} (Personal and Research Use), dated ${LICENSE_DATE}`);
    expect(LICENSE_BANNER.replace(/\s+/g, ' ')).toContain(`Version ${LICENSE_VERSION} (Personal and Research Use), dated ${LICENSE_DATE}`);
  });
});

describe('the agreement on record', () => {
  test('only a complete record counts, and only for the version in force', () => {
    expect(parseLicenseAcceptance(current)).toEqual(current);
    for (const broken of [undefined, null, 'yes', { version: '1.0' }, { ...current, acceptedAt: 'today' }, { ...current, acceptedAt: Number.NaN }]) {
      expect(parseLicenseAcceptance(broken)).toBeNull();
    }
    expect(acceptsCurrentLicense(current)).toBe(true);
    expect(acceptsCurrentLicense(null)).toBe(false);
    expect(acceptsCurrentLicense({ ...current, version: '0.9' })).toBe(false);
    expect(acceptsCurrentLicense({ ...current, date: 'January 1, 2026' })).toBe(false);
  });

  test('is kept under its own key, with the version, its date and the moment of agreement', async () => {
    const { storage, data } = memoryStorage();
    expect(await loadLicenseAcceptance(storage)).toBeNull();
    await recordLicenseAcceptance(5_000, storage);
    expect(data).toEqual({ [LICENSE_STORAGE_KEY]: { version: LICENSE_VERSION, date: LICENSE_DATE, acceptedAt: 5_000 } });
  });

  test('each agreement is the latest on record (every download of the model asks again)', async () => {
    const { storage, data } = memoryStorage({ [LICENSE_STORAGE_KEY]: current });
    await recordLicenseAcceptance(9_000, storage);
    expect(data[LICENSE_STORAGE_KEY]).toEqual({ ...current, acceptedAt: 9_000 });
  });
});

describe('the download gate', () => {
  const gate = (initial: Record<string, unknown> = {}, baseInstalled = false) => {
    const { storage, data } = memoryStorage(initial);
    const startDownload = jest.fn();
    return { deps: { storage, now: () => 7_000, baseInstalled: async () => baseInstalled, startDownload }, data, startDownload };
  };

  test('nothing is downloaded without an agreement', async () => {
    const { deps, startDownload, data } = gate();
    expect(await downloadUnderLicense(['professional'], undefined, deps)).toEqual({ ok: false, error: { code: 'LICENSE_REQUIRED' } });
    expect(startDownload).not.toHaveBeenCalled();
    expect(data).toEqual({});
  });

  test('the agreement given with the request is recorded first, then the download starts', async () => {
    const { deps, startDownload, data } = gate();
    expect(await downloadUnderLicense(['professional'], LICENSE_VERSION, deps)).toEqual({ ok: true });
    expect(data[LICENSE_STORAGE_KEY]).toEqual({ version: LICENSE_VERSION, date: LICENSE_DATE, acceptedAt: 7_000 });
    expect(startDownload).toHaveBeenCalledWith(['professional']);
    expect(await licenseAccepted(deps.storage)).toBe(true);
  });

  test('the writing model needs the agreement every time, even with one on record (a download after removing it)', async () => {
    const { deps, startDownload } = gate({ [LICENSE_STORAGE_KEY]: current }, false);
    expect(await downloadUnderLicense(['professional'], undefined, deps)).toEqual({ ok: false, error: { code: 'LICENSE_REQUIRED' } });
    expect(startDownload).not.toHaveBeenCalled();
    expect(await downloadUnderLicense(['professional'], LICENSE_VERSION, deps)).toEqual({ ok: true });
    expect(startDownload).toHaveBeenCalledTimes(1);
  });

  test('a style on top of the model needs an agreement on record, or one given with the request', async () => {
    const agreed = gate({ [LICENSE_STORAGE_KEY]: current }, true);
    expect(await downloadUnderLicense(['grammar'], undefined, agreed.deps)).toEqual({ ok: true });
    expect(agreed.startDownload).toHaveBeenCalledWith(['grammar']);
    const never = gate({}, true);
    expect(await downloadUnderLicense(['grammar'], undefined, never.deps)).toMatchObject({ ok: false });
    expect(await downloadUnderLicense(['grammar'], LICENSE_VERSION, never.deps)).toEqual({ ok: true });
  });

  test('an agreement to another version is no agreement to this one', async () => {
    const { deps, startDownload, data } = gate({ [LICENSE_STORAGE_KEY]: { ...current, version: '0.9' } }, true);
    expect(await downloadUnderLicense(['grammar'], undefined, deps)).toMatchObject({ ok: false });
    expect(await downloadUnderLicense(['grammar'], '0.9', deps)).toMatchObject({ ok: false });
    expect(startDownload).not.toHaveBeenCalled();
    expect(data[LICENSE_STORAGE_KEY]).toEqual({ ...current, version: '0.9' });
  });

  test('the refusal says what to do', () => {
    expect(describeError({ code: 'LICENSE_REQUIRED' })).toContain('Agree to the EchoMeBetter License');
  });
});

describe('messages carrying the agreement', () => {
  test('a download request may carry the version agreed to, and nothing else in its place', () => {
    expect(isUiRequest({ kind: 'ui/download-model', adapters: ['professional'] })).toBe(true);
    expect(isUiRequest({ kind: 'ui/download-model', adapters: ['professional'], acceptLicense: '1.0' })).toBe(true);
    expect(isUiRequest({ kind: 'ui/download-model', adapters: ['professional'], acceptLicense: true })).toBe(false);
    expect(isSetupRequest({ kind: 'setup/download' })).toBe(true);
    expect(isSetupRequest({ kind: 'setup/download', acceptLicense: '1.0' })).toBe(true);
    expect(isSetupRequest({ kind: 'setup/download', acceptLicense: 1 })).toBe(false);
  });

  test("a card's offer and failure say whether the agreement is still needed", () => {
    expect(isSetupEvent({ kind: 'setup/offer', totalBytes: 1, styles: ['professional'], askLicense: true })).toBe(true);
    expect(isSetupEvent({ kind: 'setup/offer', totalBytes: 1, styles: ['professional'] })).toBe(false);
    expect(isSetupEvent({ kind: 'setup/failed', error: { code: 'DOWNLOAD_FAILED' }, askLicense: false })).toBe(true);
    expect(isSetupEvent({ kind: 'setup/failed', error: { code: 'DOWNLOAD_FAILED' } })).toBe(false);
  });

  test('a page asks the service worker to open the license', () => {
    expect(isOpenLicenseRequest({ kind: 'page/open-license' })).toBe(true);
    expect(isOpenLicenseRequest({ kind: 'page/open-settings' })).toBe(false);
  });
});
