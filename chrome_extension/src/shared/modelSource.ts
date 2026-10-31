/**
 * Where the extension downloads its model from: the `url` in
 * model.source.json, at the project root.
 *
 * The URL names a folder in a KeelCore repository on Hugging Face, holding the
 * base model's model.json and the files it lists (each adapter lives in its
 * own repository, which model.json points to; see engine/manifest.ts). The
 * extension downloads them once, when the user asks, into its own private
 * storage (see worker/modelStore.ts); nothing model-sized ships in the package.
 *
 * `main` follows the repository as it is published, so a new adapter
 * reaches users without a new extension release; the commit form pins one
 * exact version. Either way every file is checked against the sha256 in
 * model.json.
 */
import source from '../../model.source.json';

/** https://huggingface.co/KeelCore/<repository>/resolve/<main or a commit>/<folder>/ */
const SOURCE_FOLDER = /^https:\/\/huggingface\.co\/KeelCore\/[A-Za-z0-9][A-Za-z0-9._-]*\/resolve\/(?:main|[0-9a-f]{40})\/(?:[A-Za-z0-9._-]+\/)*$/;

export function parseModelSourceUrl(raw: unknown): string {
  if (typeof raw !== 'string' || raw.length === 0) throw new Error('model.source.json: "url" must be a non-empty string');
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new Error(`model.source.json: "${raw}" is not a valid URL`);
  }
  if (url.protocol !== 'https:') throw new Error('model.source.json: the model must be served over https');
  if (url.search || url.hash) throw new Error('model.source.json: the URL must not have a query or fragment');
  if (!url.pathname.endsWith('/')) throw new Error('model.source.json: the URL names a folder, so it must end with "/"');
  const folder = url.toString();
  if (!SOURCE_FOLDER.test(folder) || folder.split('/').some((part) => part === '.' || part === '..')) {
    throw new Error('model.source.json: the URL must be a folder in a KeelCore repository on Hugging Face (https://huggingface.co/KeelCore/<repository>/resolve/main/<folder>/)');
  }
  return folder;
}

export const MODEL_SOURCE_URL = parseModelSourceUrl(source.url);
