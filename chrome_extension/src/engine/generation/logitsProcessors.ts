/**
 * HuggingFace's greedy-search logits processors, for a single hypothesis,
 * applied in HuggingFace's order (repetition penalty, then no-repeat n-gram).
 *
 * `sequence` is the decoder input so far, decoder-start token included --
 * the same `input_ids` HuggingFace passes its processors -- so a manifest's
 * generation settings mean exactly what they mean in `model.generate`.
 */

export interface GenerationSettings {
  readonly repetitionPenalty: number;
  readonly noRepeatNgramSize: number;
  /**
   * Token IDs the n-gram ban never applies to. The tokenizer writes every digit
   * as its own token, so `50000` legitimately repeats "0 0 0". Absent = none.
   */
  readonly noRepeatExemptTokenIds?: readonly number[];
  /** Lift the n-gram ban while a URL is being written (see UrlTracker). Absent = off. */
  readonly noRepeatExemptUrls?: boolean;
}

/**
 * A word the model is writing is a URL once it begins like one: `scheme://`,
 * `www.`, or `host.tld/` -- or is about to: the scheme or `www` written so far
 * (`https`, `https:`, `www`). SFT_adapter_rephrase/src/decoding.py holds the
 * same pattern and the same tracker; keep the two identical.
 */
const URL_WORD = /^(?:[a-z][a-z0-9+.\-]*:\/\/|www\.|(?:[a-z0-9\-]+\.)+[a-z]{2,}\/|(?:https?|ftp)(?::\/?)?$|www$)/i;
const BYTE_PIECE = /^<0x([0-9A-F]{2})>$/;

/** A vocabulary piece as plain text: `\u2581` is a space, a byte piece is its ASCII character (else U+FFFD). */
export function pieceText(piece: string): string {
  const byte = BYTE_PIECE.exec(piece);
  if (byte) {
    const value = parseInt(byte[1]!, 16);
    return value < 0x80 ? String.fromCharCode(value) : '\uFFFD';
  }
  return piece.replaceAll('\u2581', ' ');
}

/**
 * Whether the word being written is a URL, tracked token by token.
 *
 * A URL repeats its own pieces on purpose -- the host in two links, `/docs/`
 * in a path, `utm_` in each parameter -- so the n-gram ban corrupts it
 * (`utm_medium` came out `utMid`). While `inside` is true the ban is lifted;
 * the first space ends the word and restores it for the text around the URL.
 */
export class UrlTracker {
  private word = '';

  constructor(private readonly textOf: (token: number) => string) {}

  observe(token: number): void {
    const text = this.textOf(token);
    const space = text.search(/\s\S*$/u);
    this.word = space >= 0 ? text.slice(space + 1) : this.word + text;
  }

  get inside(): boolean {
    return URL_WORD.test(this.word);
  }
}

/** What the n-gram ban lets through: built once per request, read on every step. */
export interface NgramAllowance {
  readonly exempt: ReadonlySet<number>;
  /** Present when URLs are exempt; the decoder feeds it every generated token. */
  readonly urls?: UrlTracker;
  /** With `urls`: how often each n-gram (ids joined by ",") occurs in the input. */
  readonly inputNgrams?: ReadonlyMap<string, number>;
}

function countNgrams(ids: readonly number[], size: number): Map<string, number> {
  const counts = new Map<string, number>();
  for (let start = 0; start + size <= ids.length; start++) {
    const key = ids.slice(start, start + size).join(',');
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

/**
 * `piece` maps a token id to its vocabulary piece and `inputIds` is the encoder input. Recognising a URL needs the
 * first and bounding its repeats needs the second, so asking for the exemption without them is an error.
 */
export function ngramAllowance(
  settings: GenerationSettings,
  piece?: (token: number) => string | undefined,
  inputIds?: readonly number[],
): NgramAllowance {
  const exempt = new Set(settings.noRepeatExemptTokenIds ?? []);
  if (!settings.noRepeatExemptUrls) return { exempt };
  if (!piece || !inputIds) throw new Error("noRepeatExemptUrls needs the tokenizer's pieces and the input ids");
  return {
    exempt,
    urls: new UrlTracker((token) => pieceText(piece(token) ?? '')),
    inputNgrams: countNgrams(inputIds, Math.max(settings.noRepeatNgramSize, 1)),
  };
}

export function applyRepetitionPenalty(scores: Float32Array, sequence: readonly number[], penalty: number): void {
  if (penalty === 1) return;
  for (const token of new Set(sequence)) {
    const score = scores[token]!;
    scores[token] = score < 0 ? score * penalty : score / penalty;
  }
}

export function bannedNgramTokens(sequence: readonly number[], size: number, allowance?: NgramAllowance): number[] {
  if (size <= 0 || sequence.length + 1 < size) return [];
  const prefixStart = sequence.length + 1 - size;
  const seen = new Map<number, number>(); // token that completed the prefix before -> times
  for (let start = 0; start + size <= sequence.length; start++) {
    let matches = true;
    for (let k = 0; k < size - 1; k++) {
      if (sequence[start + k] !== sequence[prefixStart + k]) {
        matches = false;
        break;
      }
    }
    if (!matches) continue;
    const token = sequence[start + size - 1]!;
    if (allowance?.exempt.has(token)) continue;
    seen.set(token, (seen.get(token) ?? 0) + 1);
  }
  // Inside a URL an n-gram may repeat, but only as often as the input has it.
  const inUrl = allowance?.urls?.inside === true;
  const prefix = sequence.slice(prefixStart);
  const banned: number[] = [];
  for (const [token, times] of seen) {
    if (inUrl && times < (allowance?.inputNgrams?.get([...prefix, token].join(',')) ?? 0)) continue;
    banned.push(token);
  }
  return banned;
}

export function processLogits(
  scores: Float32Array,
  sequence: readonly number[],
  settings: GenerationSettings,
  allowance?: NgramAllowance,
): Float32Array {
  applyRepetitionPenalty(scores, sequence, settings.repetitionPenalty);
  for (const token of bannedNgramTokens(sequence, settings.noRepeatNgramSize, allowance)) scores[token] = Number.NEGATIVE_INFINITY;
  return scores;
}

/** Index of the largest score; the first one on ties, like numpy/torch argmax. */
export function argmax(scores: Float32Array): number {
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i]! > scores[best]!) best = i;
  return best;
}
