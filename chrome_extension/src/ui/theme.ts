/**
 * The popup's theme: light, dark, or whatever the system is set to. Light is
 * the default. The choice lives in its own storage key (it only concerns the
 * popup) and is applied as `data-theme` on the page's root, which the theme
 * variables in design/tailwind.cjs read; 'system' removes the attribute, so
 * the page follows the system's setting again.
 */
export const THEME_CHOICES = ['light', 'dark', 'system'] as const;

export type ThemeChoice = (typeof THEME_CHOICES)[number];

export const DEFAULT_THEME: ThemeChoice = 'light';

export const THEME_STORAGE_KEY = 'popupTheme';

/** Accept only a known choice; anything else falls back to the default. */
export function parseTheme(raw: unknown): ThemeChoice {
  return (THEME_CHOICES as readonly unknown[]).includes(raw) ? (raw as ThemeChoice) : DEFAULT_THEME;
}

export function applyTheme(theme: ThemeChoice, root: HTMLElement = document.documentElement): void {
  if (theme === 'system') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', theme);
}

/** The stored choice; the default when nothing is stored or storage can't be read. */
export async function loadTheme(storage: chrome.storage.StorageArea = chrome.storage.local): Promise<ThemeChoice> {
  try {
    const stored = await storage.get(THEME_STORAGE_KEY);
    return parseTheme(stored[THEME_STORAGE_KEY]);
  } catch {
    return DEFAULT_THEME;
  }
}

export async function saveTheme(theme: ThemeChoice, storage: chrome.storage.StorageArea = chrome.storage.local): Promise<void> {
  await storage.set({ [THEME_STORAGE_KEY]: theme });
}
