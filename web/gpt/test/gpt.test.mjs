import { test } from 'node:test';
import assert from 'node:assert/strict';
import createCharGpt from '../site/chargpt.js';

const gpt = await createCharGpt();
const vocabularySize = gpt._gpt_load();
const score = (text) => gpt.ccall('gpt_score', 'number', ['string'], [text]);

// From the last tenth of Tiny Shakespeare, which training held out.
const heldOut = 'BAPTISTA:\nA thousand thanks, Signior Gremio.\nWelcome, good Cambio.\n'
  + 'But, gentle sir, methinks you walk like a stranger:\nmay I be so bold to know the cause of your coming?\n';

function generate(prompt, seed, count) {
  gpt.ccall('gpt_reset', null, ['string', 'number'], [prompt, seed]);
  let text = '';
  for (let i = 0; i < count; i++) text += String.fromCharCode(gpt._gpt_next(0.8));
  return text;
}

test('loads the 65-character Tiny Shakespeare vocabulary', () => {
  assert.equal(vocabularySize, 65);
});

test('scores held-out Shakespeare far better than chance and than the same characters reversed', () => {
  const forward = score(heldOut);
  const reversed = score([...heldOut].reverse().join(''));
  assert.ok(forward < 1.6, `held-out loss ${forward}, chance is ln(65) = 4.17`);
  assert.ok(reversed > forward + 2, `reversed loss ${reversed}`);
});

test('writes the same text for the same seed and different text for another', () => {
  const first = generate('ROMEO:\n', 42, 200);
  assert.equal(generate('ROMEO:\n', 42, 200), first);
  assert.notEqual(generate('ROMEO:\n', 7, 200), first);
});
