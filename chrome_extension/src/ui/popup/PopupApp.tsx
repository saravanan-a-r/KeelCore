import { useState } from 'react';
import type { EngineStatus } from '../../shared/status';
import type { StyleId } from '../../shared/styles';
import { Wordmark } from '../brand/Logo';
import { PulseLogo } from '../brand/PulseLogo';
import { ModelCard } from '../components/ModelCard';
import { ShortcutAccessCard } from '../components/ShortcutAccessCard';
import { PopupStyles } from '../components/PopupStyles';
import { MACHINE, retryGpu, useGpu, useGpuProblem } from '../hooks/useCompute';
import { requestWarmUp, useEngineStatus } from '../hooks/useEngineStatus';
import {
  modelSetup,
  requestCancelDownload,
  requestDownload,
  requestRemoveAdapter,
  requestRemoveModel,
  useCatalog,
  useModelAvailability,
  type CatalogState,
  type ModelAvailability,
} from '../hooks/useModel';
import { usePopupIntent } from '../hooks/usePopupIntent';
import { useSettings } from '../hooks/useSettings';
import { useTheme } from '../hooks/useTheme';
import { needsSiteAccess, PLATFORM, releaseSiteAccess, requestSiteAccess, websiteAvailability, useSiteAccess, type WebsiteAvailability } from '../hooks/useShortcuts';
import { SettingsView } from './SettingsView';

/** Solid behind the logo, the name and the tagline (the top 62%), then fading out into the page; the status card overlaps the fade. */
const HEADER_FADE = {
  maskImage: 'linear-gradient(to bottom, #000 0%, #000 62%, transparent 100%)',
  WebkitMaskImage: 'linear-gradient(to bottom, #000 0%, #000 62%, transparent 100%)',
} as const;

export const REMOVED_NOTICE = 'Model removed. Download it again whenever you want to rewrite text.';

export interface PopupViewProps {
  readonly status: EngineStatus;
  readonly model: ModelAvailability;
  readonly catalog: CatalogState;
  /** The model was removed from this popup; say so until a new download starts. */
  readonly modelRemoved: boolean;
  readonly shortcuts: WebsiteAvailability;
  readonly holdMenu: WebsiteAvailability;
  readonly onAllowSiteAccess: () => void;
  /** "Not now" to allowing websites: what was waiting for it is turned off. */
  readonly onTurnOffWebsiteFeatures: () => void;
  readonly onOpenSettings: () => void;
  readonly onLoadModel: () => void;
  readonly onDownload: (adapters: StyleId[]) => void;
  readonly onCancelDownload: () => void;
  readonly onRetryCatalog: () => void;
  readonly onOpenGuide: () => void;
}

export function PopupView(props: PopupViewProps) {
  const { status, model, catalog, modelRemoved, shortcuts, holdMenu, onOpenGuide } = props;
  const setup = modelSetup(model.installed, catalog);
  return (
    <main className="_echo_$_w-[360px] _echo_$_bg-canvas _echo_$_text-fg">
      <header className="_echo_$_relative _echo_$_overflow-hidden _echo_$_px-4 _echo_$_pb-11 _echo_$_pt-3.5 _echo_$_text-on-accent">
        {/*
          The brand gradient under a dark scrim, solid behind the text and fading out below the tagline into
          the page's own background: the white text and the light-violet "Me" always sit on the dark part
          (WCAG AAA against the lightest pixel behind them), and there is no hard edge, in the light and the
          dark popup alike. The text must stay inside the opaque part: where the fade starts, the light page
          shows through.
        */}
        <div aria-hidden="true" className="_echo_$_absolute _echo_$_inset-0 _echo_$_bg-echo-gradient" style={HEADER_FADE}>
          <div className="_echo_$_absolute _echo_$_inset-0 _echo_$_bg-black/[0.7]" />
        </div>
        <div className="_echo_$_relative _echo_$_flex _echo_$_items-center _echo_$_gap-3">
          <span className="_echo_$_shrink-0 _echo_$_rounded-xl _echo_$_bg-on-accent/15 _echo_$_p-0.5 _echo_$_ring-1 _echo_$_ring-on-accent/30">
            <PulseLogo size={32} tone="light" />
          </span>
          <div className="_echo_$_min-w-0">
            <Wordmark tone="on-brand" className="_echo_$_block _echo_$_text-[17px] _echo_$_leading-6" />
            <p className="_echo_$_mt-0.5 _echo_$_whitespace-nowrap _echo_$_text-xs _echo_$_font-semibold _echo_$_leading-4 _echo_$_text-on-accent/90">Rewrite anything, right where you type.</p>
          </div>
          <button
            type="button"
            aria-label="Settings"
            title="Settings"
            onClick={props.onOpenSettings}
            className="_echo_$_ml-auto _echo_$_flex _echo_$_h-8 _echo_$_w-8 _echo_$_items-center _echo_$_shrink-0 _echo_$_justify-center _echo_$_rounded-lg _echo_$_text-on-accent/90 hover:_echo_$_bg-on-accent/15 hover:_echo_$_text-on-accent focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-on-accent"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z" />
            </svg>
          </button>
        </div>
      </header>

      <div className="_echo_$_space-y-3.5 _echo_$_px-4 _echo_$_pb-3 _echo_$_pt-0">
        <div className="_echo_$_mt-[-26px] _echo_$_relative">
          <ModelCard
            status={status}
            model={model}
            catalog={catalog}
            onLoadModel={props.onLoadModel}
            onDownload={props.onDownload}
            onCancelDownload={props.onCancelDownload}
            onRetryCatalog={props.onRetryCatalog}
            notice={modelRemoved ? REMOVED_NOTICE : undefined}
          />
        </div>

        {/* Until the writing model is here the setup card above is the one thing to do; the styles come after it. */}
        {model.installed && setup.catalog ? (
          <PopupStyles
            readiness={setup.readiness}
            onDownload={(adapter) => props.onDownload([adapter])}
            busy={model.download.state === 'downloading'}
            downloading={model.download.state === 'downloading' ? model.download.target.adapters : []}
          />
        ) : null}

        {needsSiteAccess(shortcuts, holdMenu) ? (
          <ShortcutAccessCard
            waiting={{ shortcuts: shortcuts === 'needs-access', holdMenu: holdMenu === 'needs-access' }}
            onAllow={props.onAllowSiteAccess}
            onNotNow={props.onTurnOffWebsiteFeatures}
          />
        ) : null}

        <footer className="_echo_$_flex _echo_$_items-center _echo_$_justify-between _echo_$_text-xs _echo_$_text-fg-muted">
          <span className="_echo_$_flex _echo_$_items-center _echo_$_gap-1.5">
            <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="_echo_$_shrink-0 _echo_$_text-better-500">
              <path d="M8 1.5 2.5 3.8v3.6c0 3.3 2.3 6.1 5.5 7.1 3.2-1 5.5-3.8 5.5-7.1V3.8L8 1.5Z" />
              <path d="m5.6 8 1.7 1.7 3.2-3.3" />
            </svg>
            On-device. Your text never leaves the browser.
          </span>
          <button type="button" onClick={onOpenGuide} className="_echo_$_font-semibold _echo_$_text-accent-fg hover:_echo_$_underline">
            Guide
          </button>
        </footer>
      </div>
    </main>
  );
}

export function PopupApp() {
  const status = useEngineStatus();
  const model = useModelAvailability();
  const [catalog, retryCatalog] = useCatalog();
  const [settings, setSettings, settingsLoaded] = useSettings();
  const [theme, setTheme] = useTheme();
  const siteAccess = useSiteAccess();
  const gpu = useGpu();
  const gpuProblem = useGpuProblem();
  const [modelRemoved, setModelRemoved] = useState(false);
  const intent = usePopupIntent();
  // Null until the user moves between pages; until then the popup opens where it was asked to.
  const [chosenPage, setPage] = useState<'home' | 'settings' | null>(null);
  const page = chosenPage ?? (intent === undefined ? null : intent ? intent.page : 'home');
  const shortcuts = websiteAvailability(settingsLoaded ? settings.shortcutsEnabled : null, siteAccess);
  const holdMenu = websiteAvailability(settingsLoaded ? settings.holdMenuEnabled : null, siteAccess);

  /** Turn website features on or off; the site access is asked for with the first and handed back with the last. */
  const setWebsiteFeatures = (next: Pick<typeof settings, 'shortcutsEnabled' | 'holdMenuEnabled'>) => {
    // Chrome only shows the permission prompt straight from a click, so ask before anything awaits.
    if (next.shortcutsEnabled || next.holdMenuEnabled) {
      if ((next.shortcutsEnabled && !settings.shortcutsEnabled) || (next.holdMenuEnabled && !settings.holdMenuEnabled)) requestSiteAccess();
    } else {
      releaseSiteAccess();
    }
    setSettings({ ...settings, ...next });
  };
  const setShortcuts = (on: boolean) => setWebsiteFeatures({ shortcutsEnabled: on, holdMenuEnabled: settings.holdMenuEnabled });
  const setHoldMenu = (on: boolean) => setWebsiteFeatures({ shortcutsEnabled: settings.shortcutsEnabled, holdMenuEnabled: on });

  // Wait the few milliseconds it takes to know where to open and which theme to wear, rather than flash the wrong page or colours.
  if (page === null || theme === null) return null;

  if (page === 'settings') {
    return (
      <SettingsView
        theme={theme}
        onThemeChange={setTheme}
        focus={chosenPage === null ? intent?.focus : undefined}
        platform={PLATFORM}
        shortcuts={shortcuts}
        onShortcutsChange={setShortcuts}
        holdMenu={holdMenu}
        onHoldMenuChange={setHoldMenu}
        onAllowSiteAccess={requestSiteAccess}
        livePreview={settingsLoaded ? settings.livePreview : null}
        onLivePreviewChange={(on) => setSettings({ ...settings, livePreview: on })}
        keepLoaded={settings.keepModelLoadedMinutes}
        onKeepLoadedChange={(minutes) => setSettings({ ...settings, keepModelLoadedMinutes: minutes })}
        performance={
          settingsLoaded
            ? {
                compute: settings,
                gpu,
                gpuProblem,
                cores: MACHINE.cores,
                crossOriginIsolated: MACHINE.crossOriginIsolated,
                status,
                onChange: (compute) => setSettings({ ...settings, ...compute }),
                onRetryGpu: retryGpu,
              }
            : null
        }
        installed={model.installed}
        onRemoveModel={async () => {
          const reply = await requestRemoveModel();
          if (reply.ok) setModelRemoved(true);
          return reply;
        }}
        onRemoveAdapter={requestRemoveAdapter}
        onBack={() => setPage('home')}
      />
    );
  }

  return (
    <PopupView
      status={status}
      model={model}
      catalog={catalog}
      modelRemoved={modelRemoved}
      shortcuts={shortcuts}
      holdMenu={holdMenu}
      onAllowSiteAccess={requestSiteAccess}
      onTurnOffWebsiteFeatures={() => setWebsiteFeatures({ shortcutsEnabled: false, holdMenuEnabled: false })}
      onOpenSettings={() => setPage('settings')}
      onLoadModel={requestWarmUp}
      onDownload={(adapters) => {
        setModelRemoved(false);
        requestDownload(adapters);
      }}
      onCancelDownload={requestCancelDownload}
      onRetryCatalog={retryCatalog}
      onOpenGuide={() => void chrome.tabs.create({ url: chrome.runtime.getURL('ui/welcome/welcome.html') })}
    />
  );
}
