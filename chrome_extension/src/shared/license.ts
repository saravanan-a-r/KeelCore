/**
 * The EchoMeBetter License (LICENSE at the project root, shipped with the
 * extension as LICENSE.txt) and the user's agreement to it.
 *
 * Nothing is downloaded until the user has agreed to the current version, and
 * every download of the writing model asks again: the service worker refuses
 * a download without it (background/licenseGate.ts).
 * The agreement is kept on this device only; nothing about it is sent
 * anywhere. When the license changes, raise the version and the date here,
 * in LICENSE and NOTICE, and every download asks for agreement again.
 */

export const LICENSE_VERSION = '1.0';
/** The version's date, as written in LICENSE ("Version Date: ..."). */
export const LICENSE_DATE = 'October 28, 2026';
export const LICENSE_NAME = 'EchoMeBetter License';

/** The license text the package ships, and the extension page that shows it. */
export const LICENSE_FILE = 'LICENSE.txt';
export const LICENSE_PAGE = 'ui/license/license.html';

export const LICENSE_STORAGE_KEY = 'licenseAcceptance';

/** The user agreed to `version` of the license (dated `date`) at `acceptedAt` (epoch ms). */
export interface LicenseAcceptance {
  readonly version: string;
  readonly date: string;
  readonly acceptedAt: number;
}

export function parseLicenseAcceptance(value: unknown): LicenseAcceptance | null {
  if (typeof value !== 'object' || value === null) return null;
  const { version, date, acceptedAt } = value as Partial<Record<keyof LicenseAcceptance, unknown>>;
  if (typeof version !== 'string' || typeof date !== 'string' || typeof acceptedAt !== 'number' || !Number.isFinite(acceptedAt)) return null;
  return { version, date, acceptedAt };
}

/** Whether `acceptance` is an agreement to the license in force now. */
export function acceptsCurrentLicense(acceptance: LicenseAcceptance | null): boolean {
  return acceptance !== null && acceptance.version === LICENSE_VERSION && acceptance.date === LICENSE_DATE;
}

export async function loadLicenseAcceptance(storage: chrome.storage.StorageArea = chrome.storage.local): Promise<LicenseAcceptance | null> {
  const stored = await storage.get(LICENSE_STORAGE_KEY);
  return parseLicenseAcceptance(stored[LICENSE_STORAGE_KEY]);
}

/** Record that the user agreed to the current license, now: each agreement (every download of the model asks again) is the latest on record. */
export async function recordLicenseAcceptance(now: number, storage: chrome.storage.StorageArea = chrome.storage.local): Promise<LicenseAcceptance> {
  const acceptance: LicenseAcceptance = { version: LICENSE_VERSION, date: LICENSE_DATE, acceptedAt: now };
  await storage.set({ [LICENSE_STORAGE_KEY]: acceptance });
  return acceptance;
}
