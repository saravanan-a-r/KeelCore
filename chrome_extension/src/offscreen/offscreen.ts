import { MODEL_SOURCE_URL } from '../shared/modelSource';
import { startBridge } from './bridge';

const worker = new Worker(new URL('../worker/inference.worker.ts', import.meta.url), { type: 'module' });

startBridge(
  worker,
  chrome.runtime,
  { modelSourceUrl: MODEL_SOURCE_URL, wasmBaseUrl: chrome.runtime.getURL('ort/') },
  navigator.storage,
);
