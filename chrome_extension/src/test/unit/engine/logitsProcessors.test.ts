import { describe, expect, test } from '@jest/globals';
import { applyRepetitionPenalty, argmax, bannedNgramTokens, ngramAllowance, pieceText, processLogits, UrlTracker } from '../../../engine/generation/logitsProcessors';

describe('repetition penalty (HuggingFace semantics)', () => {
  test('divides positive and multiplies negative scores of seen tokens', () => {
    const scores = Float32Array.from([2, -2, 2]);
    applyRepetitionPenalty(scores, [0, 1, 1], 2);
    expect(Array.from(scores)).toEqual([1, -4, 2]);
  });
});

describe('no-repeat n-gram', () => {
  test('bans the token that would complete a seen n-gram', () => {
    expect(bannedNgramTokens([0, 5, 6, 5], 2)).toEqual([6]);
    expect(bannedNgramTokens([0, 5, 6, 7, 5, 6], 3)).toEqual([7]);
  });

  test('size 1 bans every token already in the sequence', () => {
    expect(bannedNgramTokens([0, 3, 4], 1).sort()).toEqual([0, 3, 4]);
  });

  test('nothing is banned until the sequence is long enough', () => {
    expect(bannedNgramTokens([0], 3)).toEqual([]);
  });

  test('banned tokens become -Infinity', () => {
    const scores = processLogits(Float32Array.from([0, 0, 9]), [0, 2, 0], { repetitionPenalty: 1, noRepeatNgramSize: 2 });
    expect(scores[2]).toBe(Number.NEGATIVE_INFINITY);
  });
});

describe('no-repeat n-gram allowances', () => {
  const settings = { repetitionPenalty: 1, noRepeatNgramSize: 3 };

  test('without an allowance, behaviour is the plain ban', () => {
    expect(bannedNgramTokens([0, 5, 6, 7, 5, 6], 3)).toEqual([7]);
    expect(bannedNgramTokens([0, 5, 6, 7, 5, 6], 3, ngramAllowance(settings))).toEqual([7]);
  });

  test('an exempt token is never banned, others still are', () => {
    const allowance = ngramAllowance({ ...settings, noRepeatExemptTokenIds: [7] });
    expect(bannedNgramTokens([0, 5, 6, 7, 5, 6], 3, allowance)).toEqual([]);
    expect(bannedNgramTokens([0, 5, 6, 8, 5, 6], 3, allowance)).toEqual([8]);
  });

  test('a digit run can repeat its three-token window: 5 0 0 0 0', () => {
    const zero = 3;
    const plain = bannedNgramTokens([0, 4, zero, zero, zero], 3);
    expect(plain).toEqual([zero]);
    expect(bannedNgramTokens([0, 4, zero, zero, zero], 3, ngramAllowance({ ...settings, noRepeatExemptTokenIds: [zero] }))).toEqual([]);
  });

  test('banned tokens still become -Infinity through processLogits', () => {
    const allowance = ngramAllowance({ ...settings, noRepeatExemptTokenIds: [2] });
    const scores = processLogits(Float32Array.from([0, 0, 9, 9]), [0, 2, 3, 2, 3], settings, allowance);
    expect(scores[2]).toBe(9);
    const plain = processLogits(Float32Array.from([0, 0, 9, 9]), [0, 2, 3, 2, 3], settings);
    expect(plain[2]).toBe(Number.NEGATIVE_INFINITY);
  });
});

describe('URLs', () => {
  // 3 "▁https"  4 "://"  5 "ex"  6 ".com"  7 "/a"  8 "▁and"  9 "▁the"  10 "▁www"
  const PIECES: Record<number, string> = { 3: '\u2581https', 4: '://', 5: 'ex', 6: '.com', 7: '/a', 8: '\u2581and', 9: '\u2581the', 10: '\u2581www' };
  const piece = (token: number) => PIECES[token];
  const settings = { repetitionPenalty: 1, noRepeatNgramSize: 3, noRepeatExemptUrls: true };

  test('pieceText turns the space mark into a space and byte pieces into their ASCII character', () => {
    expect(pieceText('\u2581and')).toBe(' and');
    expect(pieceText('<0x2F>')).toBe('/');
    expect(pieceText('<0xC3>')).toBe('\uFFFD');
    expect(pieceText('://')).toBe('://');
  });

  test.each([
    [[3], true], // "https": a URL is about to start
    [[3, 4], true],
    [[3, 4, 5, 6, 7], true],
    [[3, 4, 5, 8], false], // a space ends the URL
    [[3, 4, 5, 8, 3, 4], true], // and a second one starts
    [[10], true], // "www": the same
    [[8], false], // "and"
    [[10, 6], true], // "www.com"
    [[5, 6, 7], true], // "ex.com/a": host.tld/ is a URL
    [[5, 6], false], // a bare domain has no path yet
    [[9, 5], false], // "the ex": ordinary words
  ])('the tracker reads %j as inside=%s', (tokens, inside) => {
    const tracker = new UrlTracker((token) => pieceText(PIECES[token] ?? ''));
    for (const token of tokens) tracker.observe(token);
    expect(tracker.inside).toBe(inside);
  });

  test('the ban is lifted inside a URL and applies again after it', () => {
    const input = [3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7]; // both links are in the input
    const allowance = ngramAllowance(settings, piece, input);
    for (const token of [3, 4, 5, 6, 7, 8, 3, 4, 5]) allowance.urls?.observe(token);
    // inside "https://ex": the repeated (4,5,6) is not banned ...
    expect(bannedNgramTokens([0, 3, 4, 5, 6, 7, 8, 3, 4, 5], 3, allowance)).toEqual([]);
    expect(bannedNgramTokens([0, 3, 4, 5, 6, 7, 8, 3, 4, 5], 3)).toEqual([6]);
    // ... but it is once the word is plain text again
    const text = ngramAllowance(settings, piece, [8, 9, 8, 9, 8, 9]);
    for (const token of [8, 9, 8, 9]) text.urls?.observe(token);
    expect(bannedNgramTokens([0, 8, 9, 8, 9, 8], 3, text)).toEqual([9]);
  });

  test('a piece repeats inside a URL only as often as the input has it', () => {
    const once = ngramAllowance(settings, piece, [3, 4, 5, 6, 7]);
    for (const token of [3, 4, 5]) once.urls?.observe(token);
    // (4,5,6) is in the input once and the output has not written it yet: nothing to ban
    expect(bannedNgramTokens([0, 3, 4, 5], 3, once)).toEqual([]);
    // the output already wrote it once and the input holds it once: banned again
    for (const token of [6, 7, 8, 3, 4, 5]) once.urls?.observe(token);
    expect(bannedNgramTokens([0, 3, 4, 5, 6, 7, 3, 4, 5], 3, once)).toEqual([6]);
    // a link the input does not contain gets no allowance
    const none = ngramAllowance(settings, piece, [9]);
    for (const token of [3, 4, 5]) none.urls?.observe(token);
    expect(bannedNgramTokens([0, 3, 4, 5, 6, 7, 3, 4, 5], 3, none)).toEqual([6]);
  });

  test('without the setting nothing changes; asking for it without a piece lookup is an error', () => {
    expect(ngramAllowance({ ...settings, noRepeatExemptUrls: false }, piece, []).urls).toBeUndefined();
    expect(ngramAllowance({ ...settings, noRepeatExemptUrls: false }).urls).toBeUndefined();
    expect(() => ngramAllowance(settings)).toThrow(/pieces/);
    expect(() => ngramAllowance(settings, piece)).toThrow(/input ids/);
  });
});

test('argmax returns the first index on ties, like torch/numpy', () => {
  expect(argmax(Float32Array.from([1, 3, 3, 2]))).toBe(1);
});
