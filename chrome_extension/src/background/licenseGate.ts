/**
 * Every download of the writing model or a style passes through here, from
 * whichever page asked (the popup, the welcome page, a website's card). The
 * pages show the agreement; this is what makes it binding, so no page can
 * start a download without it:
 *
 *   the writing model  every download of it (the first, and any after the
 *                      model was removed) needs the agreement given with
 *                      that very request
 *   a style            needs an agreement to the current license on record
 *                      (given with the model's download), or given with the request
 */
import type { UiReply } from '../shared/messages';
import { acceptsCurrentLicense, LICENSE_VERSION, loadLicenseAcceptance, recordLicenseAcceptance } from '../shared/license';
import type { StyleId } from '../shared/styles';

export interface LicenseGateDeps {
  readonly storage: chrome.storage.StorageArea;
  readonly now: () => number;
  /** Whether the writing model is on this device (a download without it brings it too). */
  readonly baseInstalled: () => Promise<boolean>;
  /** Start downloading these styles (and the base, if it is not installed). */
  readonly startDownload: (adapters: readonly StyleId[]) => void;
}

const REQUIRED: UiReply = { ok: false, error: { code: 'LICENSE_REQUIRED' } };

/**
 * Download `adapters` if the license allows it. `acceptLicense` is the
 * version the user has just agreed to on the page; it is recorded first. An
 * agreement to any other version is no agreement to this one.
 */
export async function downloadUnderLicense(adapters: readonly StyleId[], acceptLicense: string | undefined, deps: LicenseGateDeps): Promise<UiReply> {
  if (acceptLicense !== undefined) {
    if (acceptLicense !== LICENSE_VERSION) return REQUIRED;
    await recordLicenseAcceptance(deps.now(), deps.storage);
  } else if (!(await deps.baseInstalled()) || !acceptsCurrentLicense(await loadLicenseAcceptance(deps.storage))) {
    return REQUIRED;
  }
  deps.startDownload(adapters);
  return { ok: true };
}

export async function licenseAccepted(storage: chrome.storage.StorageArea): Promise<boolean> {
  return acceptsCurrentLicense(await loadLicenseAcceptance(storage));
}
