/**
 * The KeelCore License, as the extension ships it (LICENSE.txt in the
 * package): opened from the agreement next to every download, and from the
 * footers of the popup and the welcome page. It shows the exact text the
 * user agrees to, and when they agreed to it.
 */
import { useEffect, useState } from 'react';
import { acceptsCurrentLicense, LICENSE_DATE, LICENSE_FILE, LICENSE_NAME, LICENSE_VERSION, loadLicenseAcceptance, type LicenseAcceptance } from '../../shared/license';
import { Logo } from '../brand/Logo';

export const COMMERCIAL_CONTACT = 'sararavi.ar@gmail.com';

type Text = { readonly state: 'loading' } | { readonly state: 'ready'; readonly text: string } | { readonly state: 'failed' };

export interface LicenseViewProps {
  readonly text: Text;
  /** The agreement on record, if any. */
  readonly acceptance: LicenseAcceptance | null;
}

/** "You agreed to this license on 28 October 2026." */
export function agreementLine(acceptance: LicenseAcceptance | null): string | null {
  if (!acceptsCurrentLicense(acceptance)) return null;
  const when = new Date(acceptance!.acceptedAt).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
  return `You agreed to this license on ${when}.`;
}

export function LicenseView({ text, acceptance }: LicenseViewProps) {
  const agreed = agreementLine(acceptance);
  return (
    <div className="_echo_$_min-h-screen _echo_$_bg-canvas _echo_$_text-fg">
      <main className="_echo_$_mx-auto _echo_$_max-w-3xl _echo_$_px-4 _echo_$_py-10 sm:_echo_$_px-6">
        <header className="_echo_$_flex _echo_$_items-center _echo_$_gap-3">
          <Logo size={36} />
          <div>
            <h1 className="_echo_$_text-2xl _echo_$_font-bold _echo_$_tracking-tight">{LICENSE_NAME}</h1>
            <p className="_echo_$_mt-0.5 _echo_$_text-sm _echo_$_text-fg-muted">
              Version {LICENSE_VERSION} · {LICENSE_DATE}
            </p>
          </div>
        </header>

        {agreed ? (
          <p role="status" className="_echo_$_mt-6 _echo_$_rounded-lg _echo_$_bg-success-subtle _echo_$_px-3 _echo_$_py-2 _echo_$_text-sm _echo_$_font-medium _echo_$_text-success-fg">
            {agreed}
          </p>
        ) : null}

        <p className="_echo_$_mt-6 _echo_$_text-sm _echo_$_leading-6 _echo_$_text-fg-muted">
          These are the terms for using EchoMeBetter and KeelCore, the writing model it runs on. For commercial use, use by an organization, or to buy the model, write to{' '}
          <a href={`mailto:${COMMERCIAL_CONTACT}`} className="_echo_$_font-semibold _echo_$_text-accent-fg hover:_echo_$_underline">
            {COMMERCIAL_CONTACT}
          </a>
          .
        </p>

        <section aria-label="License text" className="_echo_$_mt-6 _echo_$_rounded-2xl _echo_$_border _echo_$_border-line _echo_$_bg-surface _echo_$_p-4 _echo_$_shadow-card sm:_echo_$_p-6">
          {text.state === 'ready' ? (
            // The text keeps its own line breaks and indentation, as written.
            <pre className="_echo_$_m-0 _echo_$_overflow-x-auto _echo_$_whitespace-pre-wrap _echo_$_break-words _echo_$_font-mono _echo_$_text-[13px] _echo_$_leading-6 _echo_$_text-fg">{text.text}</pre>
          ) : null}
          {text.state === 'loading' ? <p className="_echo_$_text-sm _echo_$_text-fg-muted">Loading the license…</p> : null}
          {text.state === 'failed' ? (
            <p role="alert" className="_echo_$_text-sm _echo_$_text-danger">
              The license text could not be shown. It is included with the extension as {LICENSE_FILE}; try opening this page again.
            </p>
          ) : null}
        </section>
      </main>
    </div>
  );
}

export function LicenseApp() {
  const [text, setText] = useState<Text>({ state: 'loading' });
  const [acceptance, setAcceptance] = useState<LicenseAcceptance | null>(null);
  useEffect(() => {
    let current = true;
    // The page is ui/license/license.html; the license sits at the package root.
    fetch(`../../${LICENSE_FILE}`)
      .then((response) => (response.ok ? response.text() : Promise.reject(new Error(`HTTP ${response.status}`))))
      .then(
        (body) => current && setText({ state: 'ready', text: body }),
        () => current && setText({ state: 'failed' }),
      );
    loadLicenseAcceptance().then(
      (stored) => current && setAcceptance(stored),
      () => undefined,
    );
    return () => {
      current = false;
    };
  }, []);
  return <LicenseView text={text} acceptance={acceptance} />;
}
