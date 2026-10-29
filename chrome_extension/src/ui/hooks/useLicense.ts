import { useEffect, useState } from 'react';
import { acceptsCurrentLicense, LICENSE_STORAGE_KEY, loadLicenseAcceptance, parseLicenseAcceptance } from '../../shared/license';

/**
 * Whether the user has agreed to the current KeelCore License (null until
 * it has been read). Follows an agreement given on another page or card.
 */
export function useLicenseAccepted(): boolean | null {
  const [accepted, setAccepted] = useState<boolean | null>(null);
  useEffect(() => {
    let current = true;
    void loadLicenseAcceptance().then(
      (acceptance) => current && setAccepted(acceptsCurrentLicense(acceptance)),
      // Storage that cannot be read: ask for the agreement (the service worker checks it again anyway).
      () => current && setAccepted(false),
    );
    const onChanged = (changes: Record<string, chrome.storage.StorageChange>, area: string) => {
      const change = changes[LICENSE_STORAGE_KEY];
      if (area === 'local' && change) setAccepted(acceptsCurrentLicense(parseLicenseAcceptance(change.newValue)));
    };
    chrome.storage.onChanged.addListener(onChanged);
    return () => {
      current = false;
      chrome.storage.onChanged.removeListener(onChanged);
    };
  }, []);
  return accepted;
}
