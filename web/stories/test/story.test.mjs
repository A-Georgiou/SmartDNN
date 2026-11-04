import { test } from 'node:test';
import assert from 'node:assert/strict';
import createStoryGpt from '../site/storygpt.js';

const story = await createStoryGpt();
const vocabularySize = story._story_load();
const score = (text) => story.ccall('story_score', 'number', ['string'], [text]);

// From the TinyStoriesV2-GPT4 validation file (CDLA-Sharing-1.0), which training never saw.
const heldOut = 'Once upon a time, there was a wealthy man named Tom. He had a big house near a cliff. '
  + 'Tom liked to sort his many toys into different boxes.\nOne sunny day, Tom went outside to play with his toys. '
  + 'He took them all out of their boxes and spread them on the ground. He had fun playing with his cars, dolls, and balls.';

function write(prompt, seed) {
  story.ccall('story_begin', null, ['string', 'number'], [prompt, seed]);
  const decoder = new TextDecoder();
  let text = '';
  for (let i = 0; i < 600; i++) {
    const length = story._story_next(0.8, 40);
    if (length < 0) return { text, ended: true };
    const pointer = story._story_piece();
    text += decoder.decode(story.HEAPU8.subarray(pointer, pointer + length), { stream: true });
  }
  return { text, ended: false };
}

test('loads the 2,048-token vocabulary', () => {
  assert.equal(vocabularySize, 2048);
});

test('scores a held-out story far better than chance and than its words reversed', () => {
  const forward = score(heldOut);
  const reversed = score(heldOut.split(' ').reverse().join(' '));
  assert.ok(forward < 2, `held-out loss ${forward}, chance is ln(2048) = 7.62`);
  assert.ok(reversed > forward + 4, `reversed loss ${reversed}`);
});

test('writes the same story for the same seed and a different one for another, and finishes it', () => {
  const first = write('Once upon a time', 42);
  assert.deepEqual(write('Once upon a time', 42), first);
  assert.notEqual(write('Once upon a time', 7).text, first.text);
  assert.ok(first.ended, `the story did not end: ${first.text}`);
});
