/** Light, dark, or the system's setting, for the popup. */
import type { ThemeChoice } from '../theme';

export interface ThemeSettingsProps {
  readonly theme: ThemeChoice;
  readonly onChange: (theme: ThemeChoice) => void;
}

const CHOICES: readonly { value: ThemeChoice; title: string }[] = [
  { value: 'light', title: 'Light' },
  { value: 'dark', title: 'Dark' },
  { value: 'system', title: 'System' },
];

export const THEME_HINT = "System follows your computer's light or dark setting.";

export function ThemeSettings({ theme, onChange }: ThemeSettingsProps) {
  return (
    <div>
      <div role="radiogroup" aria-label="Theme" className="_echo_$_grid _echo_$_grid-cols-3 _echo_$_gap-1 _echo_$_rounded-xl _echo_$_bg-muted _echo_$_p-1">
        {CHOICES.map((choice) => {
          const checked = choice.value === theme;
          return (
            <label
              key={choice.value}
              className={`_echo_$_cursor-pointer _echo_$_rounded-lg _echo_$_px-3 _echo_$_py-1.5 _echo_$_text-center _echo_$_transition has-[:focus-visible]:_echo_$_outline has-[:focus-visible]:_echo_$_outline-2 has-[:focus-visible]:_echo_$_outline-focus ${
                checked ? '_echo_$_bg-segment-active _echo_$_shadow-sm' : '_echo_$_text-fg-soft hover:_echo_$_bg-segment-hover'
              }`}
            >
              <input type="radio" name="theme" value={choice.value} checked={checked} onChange={() => onChange(choice.value)} className="_echo_$_sr-only" />
              <span className={`_echo_$_block _echo_$_text-sm _echo_$_font-semibold ${checked ? '_echo_$_text-accent-fg-strong' : ''}`}>{choice.title}</span>
            </label>
          );
        })}
      </div>
      <p className="_echo_$_mt-2 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-fg-muted">{THEME_HINT}</p>
    </div>
  );
}
