import { useEffect, useId, useState } from 'react';
import { describeError } from '../../shared/errors';
import { formatBytes, formatList, formatTimeLeft } from '../../shared/format';
import { styleLabel, type StyleId } from '../../shared/styles';
import { Logo } from '../../ui/brand/Logo';
import { BUTTON, GAP, offerPosition, SECONDARY_BUTTON } from './cardKit';
import type { SetupState, SetupView } from './overlayStore';

const SETUP_WIDTH = 420;
/** The card's height, close enough to decide whether it fits below the text or above it. */
const SETUP_HEIGHT = 240;
/** How long the "ready" card stays when nobody closes it. */
const DONE_MS = 30_000;

/** "This one-time download includes the Concise and Professional styles and stays on this device." */
function includes(styles: readonly StyleId[]): string {
  if (styles.length === 0) return 'This is a one-time download that stays on this device.';
  const names = formatList(styles.map(styleLabel));
  return `This one-time download includes the ${names} ${styles.length === 1 ? 'style' : 'styles'} and stays on this device.`;
}

type Downloading = Extract<SetupView, { phase: 'downloading' }>;

/** How far the download is, 0-100; checking the files counts as the end. */
function downloadPercent(view: Downloading): number {
  if (view.verifying) return 100;
  return view.totalBytes > 0 ? Math.min(100, Math.floor((view.receivedBytes / view.totalBytes) * 100)) : 0;
}

/** The words for each moment of the download. */
export function setupCopy(setup: SetupState): { title: string; body: string } {
  const { view } = setup;
  const model = setup.subject.kind === 'model';
  const name = setup.subject.kind === 'style' ? setup.subject.name : '';
  const what = model ? 'the writing model' : `the ${name} style`;
  switch (view.phase) {
    case 'checking':
      return { title: 'Preparing your download', body: 'Checking what EchoMeBetter needs…' };
    case 'offer':
      return model
        ? {
            title: 'Download the writing model',
            body: `EchoMeBetter rewrites text privately on your device, so it needs its writing model first. ${includes(view.styles)}`,
          }
        : { title: `Download the ${name} style`, body: 'Rewriting in this style needs it on your device. This is a one-time download that stays on this device.' };
    case 'downloading': {
      if (view.verifying) return { title: 'Verifying the download', body: 'Making sure every file arrived intact.' };
      const parts = [view.totalBytes > 0 ? `${formatBytes(view.receivedBytes)} of ${formatBytes(view.totalBytes)}` : 'Connecting…'];
      if (view.bytesPerSecond > 0 && view.totalBytes > 0) {
        parts.push(`${formatBytes(view.bytesPerSecond)}/s`, formatTimeLeft((view.totalBytes - view.receivedBytes) / view.bytesPerSecond));
      }
      return { title: `Downloading ${what} · ${downloadPercent(view)}%`, body: parts.join(' · ') };
    }
    case 'done':
      return model
        ? {
            title: 'Writing model installed',
            body: "You're ready to rewrite. EchoMeBetter uses your GPU when one is available, and your CPU otherwise. You can change this under Performance in Settings.",
          }
        : { title: `${name} style installed`, body: 'You can rewrite in this style now.' };
    case 'failed':
      return { title: "Download didn't finish", body: describeError(view.error) };
  }
}

interface CardAction {
  readonly label: string;
  readonly run: () => void;
}

/**
 * The card's buttons for where the download stands, the main one first. Once
 * it is done, carrying on with the rewrite comes first; settings are for those
 * who want to change the processor (after a model download only).
 */
function cardActions(setup: SetupState): CardAction[] {
  const close = { label: 'Close', run: setup.close };
  switch (setup.view.phase) {
    case 'offer':
      return [{ label: `Download · ${formatBytes(setup.view.totalBytes)}`, run: setup.download }, { ...close, label: 'Not now' }];
    case 'failed':
      return [{ label: 'Try again', run: setup.download }, close];
    case 'done': {
      const model = setup.subject.kind === 'model';
      const actions: CardAction[] = [];
      if (setup.style) actions.push({ label: `Rewrite now · ${styleLabel(setup.style)}`, run: setup.rewrite });
      if (model) actions.push({ label: 'Open settings', run: setup.openSettings });
      if (!(setup.style && model)) actions.push({ ...close, label: 'Done' });
      return actions;
    }
    default:
      return [];
  }
}

/**
 * The card a page shows when a rewrite needs the writing model, or a style,
 * that is not on this device yet. It says why, offers the download right
 * there, shows it going, and when it is done says what to do next, so the
 * user never has to leave the page they were writing on.
 */
export function SetupCard({ setup }: { setup: SetupState }) {
  const titleId = useId();
  const [hovered, setHovered] = useState(false);
  const { view } = setup;
  const say = setupCopy(setup);
  const viewport = { width: window.innerWidth, height: window.innerHeight };
  const width = Math.min(SETUP_WIDTH, viewport.width - 2 * GAP);
  const downloading = view.phase === 'downloading';
  const percent = view.phase === 'downloading' ? downloadPercent(view) : 0;

  useEffect(() => {
    if (view.phase !== 'done' || hovered) return undefined;
    const timer = setTimeout(setup.close, DONE_MS);
    return () => clearTimeout(timer);
  }, [view.phase, hovered, setup.close]);

  const actions = cardActions(setup);
  return (
    <div
      role="group"
      aria-labelledby={titleId}
      style={{ ...offerPosition(setup.anchor, viewport, width, SETUP_HEIGHT), width, contain: 'layout paint' }}
      // Keep focus (and the caret) in the user's editor when a button on the card is pressed.
      onMouseDown={(event) => event.preventDefault()}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="_echo_$_pointer-events-auto _echo_$_fixed _echo_$_flex _echo_$_animate-echo-pop-in _echo_$_overflow-hidden _echo_$_rounded-2xl _echo_$_bg-surface _echo_$_text-fg _echo_$_shadow-float _echo_$_ring-1 _echo_$_ring-sheen/15 _echo_$_will-change-transform"
    >
      <span aria-hidden="true" className={`_echo_$_w-1.5 _echo_$_shrink-0 ${view.phase === 'failed' ? '_echo_$_bg-danger' : view.phase === 'done' ? '_echo_$_bg-success' : '_echo_$_bg-accent'}`} />
      <div className="_echo_$_flex _echo_$_min-w-0 _echo_$_flex-1 _echo_$_items-start _echo_$_gap-3 _echo_$_p-4">
        <span className="_echo_$_mt-0.5 _echo_$_shrink-0">
          <Logo size={24} />
        </span>
        <div className="_echo_$_min-w-0 _echo_$_flex-1">
          {/* Each step is announced; the percent is not (the progress bar carries it). */}
          <p
            id={titleId}
            aria-live={downloading && !view.verifying ? 'off' : 'polite'}
            className="_echo_$_m-0 _echo_$_font-sans _echo_$_text-[15px] _echo_$_font-semibold _echo_$_leading-[22px] _echo_$_text-fg"
          >
            {say.title}
          </p>
          {downloading ? (
            <div
              role="progressbar"
              aria-label="Download progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={percent}
              className="_echo_$_mt-2.5 _echo_$_h-1.5 _echo_$_overflow-hidden _echo_$_rounded-full _echo_$_bg-muted"
            >
              <div
                className={`_echo_$_h-full _echo_$_rounded-full _echo_$_bg-echo-gradient _echo_$_transition-[width] _echo_$_duration-300 ${view.verifying ? '_echo_$_animate-pulse' : ''}`}
                style={{ width: `${Math.max(4, percent)}%` }}
              />
            </div>
          ) : null}
          <p className={`_echo_$_m-0 _echo_$_font-sans _echo_$_text-[14px] _echo_$_leading-5 _echo_$_text-fg-secondary ${downloading ? '_echo_$_mt-2' : '_echo_$_mt-1'}`}>{say.body}</p>
          {downloading ? (
            <p className="_echo_$_m-0 _echo_$_mt-1.5 _echo_$_font-sans _echo_$_text-[13px] _echo_$_leading-5 _echo_$_text-fg-soft">You can close this card. The download continues in the background.</p>
          ) : null}
          {actions.length > 0 ? (
            <div className="_echo_$_mt-3.5 _echo_$_flex _echo_$_flex-wrap _echo_$_items-center _echo_$_gap-2">
              {actions.map((action, index) => (
                <button key={action.label} type="button" onClick={action.run} className={index === 0 ? BUTTON : SECONDARY_BUTTON}>
                  {action.label}
                </button>
              ))}
            </div>
          ) : null}
        </div>
        <button
          type="button"
          aria-label="Close"
          onClick={setup.close}
          className="_echo_$_m-[-4px] _echo_$_flex _echo_$_h-7 _echo_$_w-7 _echo_$_shrink-0 _echo_$_cursor-pointer _echo_$_items-center _echo_$_justify-center _echo_$_rounded-md _echo_$_border-0 _echo_$_bg-transparent _echo_$_p-0 _echo_$_text-fg-soft hover:_echo_$_bg-sheen/10 hover:_echo_$_text-fg"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
      </div>
    </div>
  );
}
