import { useCallback, useEffect, useLayoutEffect, useState } from 'react';
import { applyTheme, loadTheme, saveTheme, type ThemeChoice } from '../theme';

/** The popup's theme (null until it has been read from storage), and a way to change it. The page takes the theme before it is painted. */
export function useTheme(): [ThemeChoice | null, (next: ThemeChoice) => void] {
  const [theme, setTheme] = useState<ThemeChoice | null>(null);
  useEffect(() => {
    let current = true;
    void loadTheme().then((stored) => {
      if (current) setTheme(stored);
    });
    return () => {
      current = false;
    };
  }, []);
  useLayoutEffect(() => {
    if (theme !== null) applyTheme(theme);
  }, [theme]);
  const update = useCallback((next: ThemeChoice) => {
    setTheme(next);
    void saveTheme(next);
  }, []);
  return [theme, update];
}
