/**
 * The EchoMeBetter mark: a voice (the dot) and its echo (two arcs), with a
 * spark for "better", on the violet-to-mint brand tile.
 *
 * The geometry is shared with EchoLoader so the loader reads as the logo
 * coming alive, and with the toolbar icons rendered from it.
 */
import { useId } from 'react';
import { brand } from '../../design/tokens.cjs';

export const MARK = {
  dot: { cx: 20, cy: 32, r: 5 },
  innerArc: 'M27.7 22.8 A12 12 0 0 1 27.7 41.2',
  outerArc: 'M33.5 15.9 A21 21 0 0 1 33.5 48.1',
  spark: 'M48 9 C48.6 13.6 50.4 15.4 55 16 C50.4 16.6 48.6 18.4 48 23 C47.4 18.4 45.6 16.6 41 16 C45.6 15.4 47.4 13.6 48 9Z',
} as const;

export function Logo({ size = 32, title = 'EchoMeBetter' }: { size?: number; title?: string }) {
  const gradient = useId();
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" role="img" aria-label={title}>
      <defs>
        <linearGradient id={gradient} x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor={brand['logo-start']} />
          <stop offset="0.55" stopColor={brand['logo-middle']} />
          <stop offset="1" stopColor={brand['logo-end']} />
        </linearGradient>
      </defs>
      <rect x="2" y="2" width="60" height="60" rx="16" fill={`url(#${gradient})`} />
      <circle {...MARK.dot} fill={brand['logo-mark']} />
      <path d={MARK.innerArc} stroke={brand['logo-mark']} strokeWidth="4.5" strokeLinecap="round" fill="none" />
      <path d={MARK.outerArc} stroke={brand['logo-mark']} strokeOpacity="0.85" strokeWidth="4.5" strokeLinecap="round" fill="none" />
      <path d={MARK.spark} fill={brand['logo-spark']} />
    </svg>
  );
}

/**
 * The product name, with "Me" picked out in violet: deep violet on light
 * backgrounds, light violet on dark ones (both pass WCAG AAA).
 *   tone 'page'      on the page's own background; follows its theme.
 *   tone 'on-brand'  on the violet brand header, which is dark in every theme.
 */
export function Wordmark({ className = '', tone = 'page' }: { className?: string; tone?: 'page' | 'on-brand' }) {
  const onBrand = tone === 'on-brand';
  return (
    <span className={`_echo_$_font-semibold _echo_$_tracking-tight ${onBrand ? '_echo_$_text-on-accent' : '_echo_$_text-fg'} ${className}`}>
      Echo<span className={onBrand ? '_echo_$_text-echo-300' : '_echo_$_text-brand-name'}>Me</span>Better
    </span>
  );
}
