/**
 * A text too long for one pass is rewritten in chunks. Text the prompt builder refuses must still fail
 * straight away, before any chunk goes through the model (as it does for a short text).
 */
import { afterAll, beforeAll, expect, jest, test } from '@jest/globals';
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
