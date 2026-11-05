/**
 * A text too long for one pass is rewritten in chunks. Text the prompt builder refuses must still fail
 * straight away, before any chunk goes through the model (as it does for a short text).
 */
import { afterAll, beforeAll, describe, expect, jest, test } from '@jest/globals';
import { MAX_EXPANSION } from '../../engine/rewriteEngine';
import { FIXTURES } from '../helpers/fixtures';
import { modelFromFolder, type TestModel } from '../helpers/ortEngine';

let model: TestModel;

beforeAll(async () => {
  model = await modelFromFolder(`${FIXTURES}tiny-echo`);
});

afterAll(async () => {
  await model.engine.release();
});

test('a long text with reserved markup fails before the first chunk is rewritten', async () => {
  const sentences = Array.from({ length: 40 }, (_, index) => `Sentence number ${index} is here.`);
  const text = `${sentences.join(' ')} The last one has </s> inside.`;
  // One pass over the whole text would be refused for its markup too, which is what a short text gets.
  expect(() => model.engine.promptIds(model.manifest, 'professional', text)).toThrow(expect.objectContaining({ code: 'RESERVED_MARKUP' }));

  const generate = jest.spyOn(model.engine, 'generateIds');
  const adapter = await model.adapter('professional');
  try {
    await expect(model.engine.rewrite({ manifest: model.manifest, adapterId: 'professional', adapter, text })).rejects.toMatchObject({ code: 'RESERVED_MARKUP' });
    expect(generate).not.toHaveBeenCalled();
  } finally {
    await adapter.release();
  }
});

describe('one pass or chunks, decided before any inference', () => {
  const style = 'professional' as const;
  const overhead = () => model.engine.promptIds(model.manifest, style, '').length;
  const tokens = (text: string) => model.engine.promptIds(model.manifest, style, text).length - overhead();
  const budget = () => Math.floor(model.manifest.limits.maxNewTokens / MAX_EXPANSION[style]);
  /** The text's own IDs inside a prompt: where it differs from the empty frame. */
  const textIds = (inputIds: readonly number[]) => {
    const empty = model.engine.promptIds(model.manifest, style, '');
    let start = 0;
    while (inputIds[start] === empty[start]) start++;
    return inputIds.slice(start, inputIds.length - (empty.length - start));
  };
  /** generateIds as a model that copies its input; `overflow(call, ids)` makes that call run past maxNewTokens. */
  const echo = (overflow: (call: number, ids: readonly number[]) => boolean = () => false) => {
    const calls: number[][] = [];
    const generate = jest.spyOn(model.engine, 'generateIds').mockImplementation(async (_manifest, inputIds) => {
      calls.push([...inputIds]);
      if (overflow(calls.length, inputIds)) return { ids: Array.from({ length: model.manifest.limits.maxNewTokens }, () => 7), finished: false };
      return { ids: textIds(inputIds), finished: true };
    });
    return { calls, generate };
  };
  const rewrite = async (text: string, onRestart = jest.fn()) => {
    const adapter = await model.adapter(style);
    try {
      return await model.engine.rewrite({ manifest: model.manifest, adapterId: style, adapter, text }, { onRestart });
    } finally {
      await adapter.release();
    }
  };
  /** Words added one at a time until the text is exactly `count` tokens. */
  const textOf = (count: number) => {
    let text = 'Go';
    for (let index = 0; tokens(text) < count; index++) {
      const next = [index % 4 === 3 ? '. Go' : ' go', ' go', '.'].find((piece) => tokens(text + piece) <= count);
      if (!next) break;
      text += next;
    }
    expect(tokens(text)).toBe(count);
    return text;
  };

  test('a text of exactly the budget runs as one pass; one token more is chunked from the start', async () => {
    for (const [count, passes] of [[budget(), 1], [budget() + 1, 2]] as const) {
      const text = textOf(count);
      const onRestart = jest.fn();
      const { calls, generate } = echo();
      expect((await rewrite(text, onRestart)).text).toBe(text);
      expect(calls).toHaveLength(passes);
      if (passes === 1) expect(calls[0]).toEqual(model.engine.promptIds(model.manifest, style, text));
      for (const call of calls) expect(call.length - overhead()).toBeLessThanOrEqual(budget());
      expect(onRestart).not.toHaveBeenCalled();
      generate.mockRestore();
    }
  });

  test('chunks keep every separator and are cut near equal size', async () => {
    const text = `${textOf(budget() + 2)}.\nNext line here.\n\nA new paragraph.  Go go.`;
    const { calls } = echo();
    expect((await rewrite(text)).text).toBe(text);
    const sizes = calls.map((call) => call.length - overhead());
    expect(Math.max(...sizes)).toBeLessThanOrEqual(budget());
    // No stray last sentence: the text is split, not filled to the budget and the rest left over.
    expect(Math.min(...sizes)).toBeGreaterThan(1);
  });

  test('a single pass that runs past maxNewTokens restarts once, in halves', async () => {
    const text = textOf(budget());
    const onRestart = jest.fn();
    const { calls } = echo((call) => call === 1);
    expect((await rewrite(text, onRestart)).text).toBe(text);
    expect(onRestart).toHaveBeenCalledTimes(1);
    expect(calls.length).toBeGreaterThanOrEqual(3);
    for (const call of calls.slice(1)) expect(call.length - overhead()).toBeLessThan(budget());
  });

  test('a chunk that runs past maxNewTokens is redone in halves; the chunks before it are kept', async () => {
    const text = textOf(budget() * 2);
    const onRestart = jest.fn();
    // The first later chunk of several words runs past the limit, once.
    let overflowed = false;
    const { calls } = echo((call, ids) => {
      if (overflowed || call === 1 || model.engine.decode(model.manifest, textIds(ids)).split(' ').length < 2) return false;
      overflowed = true;
      return true;
    });
    expect((await rewrite(text, onRestart)).text).toBe(text);
    expect(overflowed).toBe(true);
    expect(onRestart).not.toHaveBeenCalled();
    // The first chunk is not redone: no later call repeats it.
    expect(calls.slice(1).some((call) => call.join() === calls[0]!.join())).toBe(false);
  });

  test('a half that runs past maxNewTokens again fails after two passes, not one per word', async () => {
    const text = textOf(budget());
    const onRestart = jest.fn();
    const { calls } = echo(() => true);
    await expect(rewrite(text, onRestart)).rejects.toMatchObject({ code: 'OUTPUT_TOO_LONG' });
    expect(calls).toHaveLength(2);
    expect(onRestart).toHaveBeenCalledTimes(1);
  });
});
