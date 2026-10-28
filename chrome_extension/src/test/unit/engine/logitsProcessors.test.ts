import { describe, expect, test } from '@jest/globals';
import { applyRepetitionPenalty, argmax, bannedNgramTokens, ngramAllowance, processLogits } from '../../../engine/generation/logitsProcessors';

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

test('argmax returns the first index on ties, like torch/numpy', () => {
  expect(argmax(Float32Array.from([1, 3, 3, 2]))).toBe(1);
});
