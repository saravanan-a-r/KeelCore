/** @jest-environment jsdom */
import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, render, renderHook, screen, waitFor } from '@testing-library/react';
import { LICENSE_DATE, LICENSE_STORAGE_KEY, LICENSE_VERSION } from '../../../shared/license';
import { useLicenseAccepted } from '../../../ui/hooks/useLicense';
import { agreementLine, LicenseApp, LicenseView } from '../../../ui/license/LicenseApp';

const agreement = { version: LICENSE_VERSION, date: LICENSE_DATE, acceptedAt: Date.UTC(2026, 9, 28, 12) };

let stored: Record<string, unknown>;
let listeners: Array<(changes: Record<string, chrome.storage.StorageChange>, area: string) => void>;

beforeEach(() => {
  stored = {};
  listeners = [];
  (globalThis as { chrome?: unknown }).chrome = {
    storage: {
      local: { get: jest.fn(async (key: string) => (key in stored ? { [key]: stored[key] } : {})) },
      onChanged: {
        addListener: (listener: (typeof listeners)[number]) => listeners.push(listener),
        removeListener: (listener: (typeof listeners)[number]) => void (listeners = listeners.filter((other) => other !== listener)),
      },
    },
  };
});

afterEach(() => {
  delete (globalThis as { fetch?: unknown }).fetch;
});

describe('the license page', () => {
  test('shows the text as written, with its version and date, and the contact for commercial use', () => {
    render(<LicenseView text={{ state: 'ready', text: 'EchoMeBetter License\n\n1. ACCEPTANCE\n\n1.1 This license…' }} acceptance={null} />);
    expect(screen.getByRole('heading', { name: 'EchoMeBetter License' })).toBeInTheDocument();
    expect(screen.getByText(`Version ${LICENSE_VERSION} · ${LICENSE_DATE}`)).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'License text' })).toHaveTextContent('1. ACCEPTANCE');
    expect(screen.getByRole('link', { name: 'sararavi.ar@gmail.com' })).toHaveAttribute('href', 'mailto:sararavi.ar@gmail.com');
    // Nobody has agreed yet: no agreement is claimed.
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  test('says when the user agreed to this version, and only to this version', () => {
    expect(agreementLine(agreement)).toMatch(/^You agreed to this license on .*2026\.$/);
    expect(agreementLine({ ...agreement, version: '0.9' })).toBeNull();
    expect(agreementLine(null)).toBeNull();
  });

  test('loads the license shipped with the extension, and says so if it cannot', async () => {
    const fetch = jest.fn(async (_url: string) => ({ ok: true, text: async () => 'THE LICENSE TEXT' }));
    Object.assign(globalThis, { fetch });
    stored[LICENSE_STORAGE_KEY] = agreement;
    render(<LicenseApp />);
    expect(await screen.findByText('THE LICENSE TEXT')).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith('../../LICENSE.txt');
    expect(await screen.findByRole('status')).toHaveTextContent('You agreed to this license on');
  });

  test('a license that cannot be read is said plainly', async () => {
    Object.assign(globalThis, { fetch: jest.fn(async () => ({ ok: false, status: 404, text: async () => '' })) });
    render(<LicenseApp />);
    expect(await screen.findByRole('alert')).toHaveTextContent('included with the extension as LICENSE.txt');
  });
});

describe('useLicenseAccepted', () => {
  test('reads the agreement on record, and follows one given on another page', async () => {
    const { result, unmount } = renderHook(() => useLicenseAccepted());
    expect(result.current).toBeNull();
    await waitFor(() => expect(result.current).toBe(false));
    act(() => listeners.forEach((listener) => listener({ [LICENSE_STORAGE_KEY]: { newValue: agreement } as chrome.storage.StorageChange }, 'local')));
    expect(result.current).toBe(true);
    // Other storage areas and keys are not this.
    act(() => listeners.forEach((listener) => listener({ [LICENSE_STORAGE_KEY]: { newValue: undefined } as chrome.storage.StorageChange }, 'sync')));
    expect(result.current).toBe(true);
    unmount();
    expect(listeners).toHaveLength(0);
  });

  test('an older agreement does not count; unreadable storage asks for it', async () => {
    stored[LICENSE_STORAGE_KEY] = { ...agreement, version: '0.9' };
    const older = renderHook(() => useLicenseAccepted());
    await waitFor(() => expect(older.result.current).toBe(false));
    (chrome.storage.local.get as jest.Mock).mockImplementationOnce(async () => {
      throw new Error('no storage');
    });
    const broken = renderHook(() => useLicenseAccepted());
    await waitFor(() => expect(broken.result.current).toBe(false));
  });
});
