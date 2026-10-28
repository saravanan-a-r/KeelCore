/**
 * The agreement to the EchoMeBetter License, asked for right above a
 * download button until the user has given it once (for the current
 * version). The box starts unticked and the download stays disabled until it
 * is ticked; the link opens the full license in a tab of its own.
 */
import { LICENSE_NAME, LICENSE_PAGE } from '../../shared/license';

/** The license page, from any of the extension's pages (they all sit two folders down: ui/<page>/<page>.html). */
export const LICENSE_HREF = `../../${LICENSE_PAGE}`;

export interface LicenseConsentProps {
  readonly id: string;
  readonly checked: boolean;
  readonly onChange: (checked: boolean) => void;
  readonly className?: string;
}

export function LicenseConsent({ id, checked, onChange, className = '' }: LicenseConsentProps) {
  return (
    // A link inside a label does not tick the box: reading the license and agreeing to it stay two separate acts.
    <label htmlFor={id} className={`_echo_$_flex _echo_$_cursor-pointer _echo_$_items-start _echo_$_gap-2 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-fg-secondary ${className}`}>
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="_echo_$_mt-0.5 _echo_$_h-4 _echo_$_w-4 _echo_$_shrink-0 _echo_$_cursor-pointer _echo_$_accent-accent focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-offset-2 focus-visible:_echo_$_outline-focus"
      />
      <span>
        I agree to the{' '}
        <a href={LICENSE_HREF} target="_blank" rel="noopener" className="_echo_$_font-semibold _echo_$_text-accent-fg _echo_$_underline _echo_$_underline-offset-2 hover:_echo_$_no-underline">
          {LICENSE_NAME}
        </a>
      </span>
    </label>
  );
}

/** Said on a download button kept disabled until the agreement is ticked. */
export const AGREE_FIRST = `Agree to the ${LICENSE_NAME} first`;
