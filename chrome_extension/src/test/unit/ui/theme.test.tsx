/** @jest-environment jsdom */
import { describe, expect, jest, test } from '@jest/globals';
import { fireEvent, render, screen } from '@testing-library/react';
import { themeVariables } from '../../../design/tailwind.cjs';
import { ThemeSettings } from '../../../ui/components/ThemeSettings';
import { applyTheme, DEFAULT_THEME, loadTheme, parseTheme, saveTheme, THEME_STORAGE_KEY } from '../../../ui/theme';

const storageWith = (data: Record<string, unknown>) =>
  ({ get: jest.fn(async (key: string) => (key in data ? { [key]: data[key] } : {})), set: jest.fn(async () => undefined) }) as unknown as chrome.storage.StorageArea;

describe('popup theme', () => {
  test('light is the default, and only known choices are accepted', () => {
    expect(DEFAULT_THEME).toBe('light');
    expect(parseTheme(undefined)).toBe('light');
    expect(parseTheme('purple')).toBe('light');
    for (const choice of ['light', 'dark', 'system']) expect(parseTheme(choice)).toBe(choice);
  });

  test('a stored choice is read back, and nothing stored means light', async () => {
    expect(await loadTheme(storageWith({ [THEME_STORAGE_KEY]: 'dark' }))).toBe('dark');
    expect(await loadTheme(storageWith({}))).toBe('light');
  });

  test('when storage cannot be read the popup still gets a theme', async () => {
    const broken = { get: jest.fn(async () => { throw new Error('no storage'); }) } as unknown as chrome.storage.StorageArea;
    expect(await loadTheme(broken)).toBe('light');
  });

  test('a choice is saved under its own key, apart from the other settings', async () => {
    const storage = storageWith({});
    await saveTheme('system', storage);
    expect(storage.set).toHaveBeenCalledWith({ popupTheme: 'system' });
  });

  test('light and dark are set on the page root; system hands the page back to the system', () => {
    const root = document.createElement('html');
    applyTheme('dark', root);
    expect(root.getAttribute('data-theme')).toBe('dark');
    applyTheme('light', root);
    expect(root.getAttribute('data-theme')).toBe('light');
    applyTheme('system', root);
    expect(root.hasAttribute('data-theme')).toBe(false);
  });

  test('the theme variables let data-theme beat the system setting, and only on pages that follow the system', () => {
    const run = (scheme: 'system' | 'inverse') => {
      const base: Record<string, Record<string, string>>[] = [];
      (themeVariables({ selector: ':root', scheme }) as unknown as { handler: (api: unknown) => void }).handler({ addBase: (rules: Record<string, Record<string, string>>) => base.push(rules), addUtilities: () => undefined });
      return base.flatMap((rules) => Object.keys(rules));
    };
    expect(run('system')).toEqual(expect.arrayContaining([':root', '@media (prefers-color-scheme: dark)', ':root[data-theme="light"]', ':root[data-theme="dark"]']));
    expect(run('inverse').filter((selector) => selector.includes('data-theme'))).toEqual([]);
  });
});

describe('theme setting', () => {
  test('offers light, dark and system, with the current one checked', () => {
    render(<ThemeSettings theme="light" onChange={jest.fn()} />);
    expect(screen.getByRole('radiogroup', { name: 'Theme' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Light' })).toBeChecked();
    expect(screen.getByRole('radio', { name: 'Dark' })).not.toBeChecked();
    expect(screen.getByRole('radio', { name: 'System' })).not.toBeChecked();
  });

  test('choosing one reports it', () => {
    const onChange = jest.fn();
    render(<ThemeSettings theme="light" onChange={onChange} />);
    fireEvent.click(screen.getByRole('radio', { name: 'Dark' }));
    expect(onChange).toHaveBeenCalledWith('dark');
    fireEvent.click(screen.getByRole('radio', { name: 'System' }));
    expect(onChange).toHaveBeenCalledWith('system');
  });
});
