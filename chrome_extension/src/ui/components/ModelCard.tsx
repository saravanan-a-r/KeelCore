/**
 * The model's card, the same on the popup and the welcome page: a download
 * in progress (or one that stopped) first, then the model's status once it
 * is installed, and the first-run setup before that.
 */
import type { StyleId } from '../../shared/styles';
import type { EngineStatus } from '../../shared/status';
import type { CatalogState, ModelAvailability } from '../hooks/useModel';
import { ModelDownloadCard } from './ModelDownloadCard';
import { ModelSetupCard } from './ModelSetupCard';
import { StatusCard } from './StatusCard';

export interface ModelCardProps {
  readonly status: EngineStatus;
  readonly model: ModelAvailability;
  readonly catalog: CatalogState;
  readonly onLoadModel: () => void;
  /** `agreed`: the user has just ticked the license agreement. */
  readonly onDownload: (adapters: StyleId[], agreed?: boolean) => void;
  /** Whether the user has agreed to the current license; null until it has been read (the card waits for it). */
  readonly licenseAccepted: boolean | null;
  readonly onCancelDownload: () => void;
  readonly onRetryCatalog: () => void;
  /** Shown on the setup card, e.g. right after the model was removed. */
  readonly notice?: string;
}

export function ModelCard({ status, model, catalog, onLoadModel, onDownload, licenseAccepted, onCancelDownload, onRetryCatalog, notice }: ModelCardProps) {
  if (!model.known || licenseAccepted === null) return null;
  if (model.download.state !== 'idle') {
    return (
      <ModelDownloadCard
        download={model.download}
        onRetry={(target, agreed) => onDownload([...target.adapters], agreed)}
        onCancel={onCancelDownload}
        licenseAccepted={licenseAccepted}
      />
    );
  }
  if (model.installed) return <StatusCard status={status} onLoad={onLoadModel} />;
  return <ModelSetupCard catalog={catalog} onDownload={onDownload} onRetryCatalog={onRetryCatalog} notice={notice} />;
}
