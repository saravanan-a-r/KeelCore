import { describe, expect, jest, test } from '@jest/globals';
import { greedyGenerate, type Seq2SeqRunner } from '../../../engine/generation/greedyDecoder';

const EOS = 1;

/** A runner that emits a scripted token per step and records what it was fed. */
function scriptedRunner(script: number[], vocab = 10) {
  const fed: { token: number; cache: number }[] = [];
  const release = jest.fn();
  const runner: Seq2SeqRunner<string, number> = {
    encode: async () => 'memory',
    initialCache: () => 0,
    step: async (token, memory, cache) => {
      expect(memory).toBe('memory');
      fed.push({ token, cache });
      const logits = new Float32Array(vocab);
      logits[script[cache] ?? EOS] = 1;
      return { logits, cache: cache + 1 };
    },
    release,
  };
  return { runner, fed, release };
}

const options = { decoderStartId: 0, eosId: EOS, maxNewTokens: 8, repetitionPenalty: 1, noRepeatNgramSize: 0 };

describe('greedyGenerate', () => {
  test('feeds each token back with the previous step cache and stops at EOS (excluded)', async () => {
    const { runner, fed, release } = scriptedRunner([4, 5, EOS]);
    const result = await greedyGenerate(runner, [7, 8], options);
    expect(result).toEqual({ ids: [4, 5], finished: true });
    expect(fed).toEqual([
      { token: 0, cache: 0 },
      { token: 4, cache: 1 },
      { token: 5, cache: 2 },
    ]);
    expect(release).toHaveBeenCalledTimes(1);
  });

  test('reports unfinished output when maxNewTokens is reached', async () => {
    const { runner } = scriptedRunner([4, 5, 6, 7]);
    expect(await greedyGenerate(runner, [7], { ...options, maxNewTokens: 3 })).toEqual({ ids: [4, 5, 6], finished: false });
  });

  test('cancellation stops before the next step and still releases resources', async () => {
    const { runner, release, fed } = scriptedRunner([4, 5, 6, 7]);
    const controller = new AbortController();
    const pending = greedyGenerate(runner, [7], {
      ...options,
      signal: controller.signal,
      onToken: (count) => count === 2 && controller.abort(),
    });
    await expect(pending).rejects.toMatchObject({ code: 'CANCELLED' });
    expect(fed).toHaveLength(2);
    expect(release).toHaveBeenCalledTimes(1);
  });

  test('an already-aborted signal never touches the model', async () => {
    const { runner, fed } = scriptedRunner([4]);
    const controller = new AbortController();
    controller.abort();
    await expect(greedyGenerate(runner, [7], { ...options, signal: controller.signal })).rejects.toMatchObject({ code: 'CANCELLED' });
    expect(fed).toHaveLength(0);
  });
});

describe('greedyGenerate with exempt tokens', () => {
  const ban3 = { ...options, maxNewTokens: 12, noRepeatNgramSize: 3 };

  test('a repeated digit is blocked by the plain ban and written when its token is exempt', async () => {
    const script = [7, 7, 7, 7, 7, EOS];
    const plain = await greedyGenerate(scriptedRunner(script).runner, [9], ban3);
    expect(plain.ids.slice(0, 5)).not.toEqual([7, 7, 7, 7, 7]);
    const exempt = await greedyGenerate(scriptedRunner(script).runner, [9], { ...ban3, noRepeatExemptTokenIds: [7] });
    expect(exempt).toEqual({ ids: [7, 7, 7, 7, 7], finished: true });
  });
});

describe('greedyGenerate inside URLs', () => {
  // 3 "▁https"  4 "://"  5 "ex"  6 ".com"  7 "/a"  8 "▁and"
  const PIECES: Record<number, string> = { 3: '\u2581https', 4: '://', 5: 'ex', 6: '.com', 7: '/a', 8: '\u2581and' };
  const ban3 = { ...options, maxNewTokens: 14, noRepeatNgramSize: 3, piece: (token: number) => PIECES[token] };
  const twice = [3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, EOS];

  test('a second link may repeat the first one\'s pieces when URLs are exempt', async () => {
    const input = twice.slice(0, -1); // the input holds both links
    const plain = await greedyGenerate(scriptedRunner(twice).runner, input, ban3);
    expect(plain.ids).not.toEqual(twice.slice(0, -1));
    const exempt = await greedyGenerate(scriptedRunner(twice).runner, input, { ...ban3, noRepeatExemptUrls: true });
    expect(exempt).toEqual({ ids: twice.slice(0, -1), finished: true });
  });

  test('a link repeats only as often as the input has it', async () => {
    const once = [3, 4, 5, 6, 7];
    const runaway = [...once, 6, 7, 6, 7, 6, 7, 6, 7, EOS];
    const result = await greedyGenerate(scriptedRunner(runaway).runner, once, { ...ban3, noRepeatExemptUrls: true });
    expect(result.ids).not.toEqual(runaway.slice(0, -1));
    expect(result.ids.slice(0, 5)).toEqual(once);
  });

  test('a link the input does not have gets no allowance', async () => {
    const result = await greedyGenerate(scriptedRunner(twice).runner, [9], { ...ban3, noRepeatExemptUrls: true });
    expect(result.ids).not.toEqual(twice.slice(0, -1));
  });

  test('plain text is still protected from loops when URLs are exempt', async () => {
    const loop = [8, 9, 8, 9, 8, 9, 8, 9, EOS];
    const a = await greedyGenerate(scriptedRunner(loop).runner, loop, { ...ban3, noRepeatExemptUrls: true });
    const b = await greedyGenerate(scriptedRunner(loop).runner, loop, ban3);
    expect(a).toEqual(b);
  });
});
