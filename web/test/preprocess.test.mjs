import { test } from 'node:test';
import assert from 'node:assert/strict';
import { toMnistPixels } from '../site/preprocess.js';

const SIZE = 280;

function padWithRect(x0, y0, x1, y1) {
  const rgba = new Uint8ClampedArray(SIZE * SIZE * 4);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) rgba.fill(255, (y * SIZE + x) * 4, (y * SIZE + x + 1) * 4);
  }
  return rgba;
}

test('a blank pad gives no input', () => {
  assert.equal(toMnistPixels(new Uint8ClampedArray(SIZE * SIZE * 4), SIZE, SIZE), null);
});

test('an off-centre stroke is scaled into 20 x 20 and centred by mass', () => {
  const pixels = toMnistPixels(padWithRect(10, 15, 30, 115), SIZE, SIZE);  // A 20 x 100 bar near the top-left corner.
  let mass = 0, massX = 0, massY = 0;
  const rows = new Set(), cols = new Set();
  pixels.forEach((value, i) => {
    if (value === 0) return;
    rows.add(Math.floor(i / 28));
    cols.add(i % 28);
    mass += value;
    massX += value * (i % 28);
    massY += value * Math.floor(i / 28);
  });
  assert.equal(rows.size, 20);
  assert.equal(cols.size, 4);
  assert.ok(Math.abs(massX / mass - 13.5) <= 0.5);
  assert.ok(Math.abs(massY / mass - 13.5) <= 0.5);
});

test('the WebAssembly model reads a drawn vertical stroke as a 1', async () => {
  const { default: createSmartDNN } = await import('../site/smartdnn.js');
  const smartdnn = await createSmartDNN();
  smartdnn._load_model();
  const input = smartdnn._malloc(784 * Float32Array.BYTES_PER_ELEMENT);
  smartdnn.HEAPF32.set(toMnistPixels(padWithRect(130, 40, 152, 240), SIZE, SIZE), input / 4);
  const offset = smartdnn._predict(input) / 4;
  const probabilities = smartdnn.HEAPF32.slice(offset, offset + 10);
  assert.ok(Math.abs(probabilities.reduce((sum, p) => sum + p, 0) - 1) < 1e-4);
  assert.equal(probabilities.indexOf(Math.max(...probabilities)), 1);
});
