import createSmartDNN from './smartdnn.js';
import { toMnistPixels } from './preprocess.js';

const pad = document.getElementById('pad');
const padCtx = pad.getContext('2d', { willReadFrequently: true });
const seenCtx = document.getElementById('seen').getContext('2d');
const guessEl = document.getElementById('guess');
const confidenceEl = document.getElementById('confidence');
const statusEl = document.getElementById('status');

const rows = Array.from({ length: 10 }, (_, digit) => {
  const li = document.createElement('li');
  li.innerHTML = `<span>${digit}</span><div class="bar"><div class="fill"></div></div><span class="percent">0.0%</span>`;
  document.getElementById('bars').append(li);
  return { li, fill: li.querySelector('.fill'), percent: li.querySelector('.percent') };
});

let smartdnn;
let inputPtr;
try {
  smartdnn = await createSmartDNN();
  smartdnn._load_model();
  inputPtr = smartdnn._malloc(784 * Float32Array.BYTES_PER_ELEMENT);
  statusEl.textContent = 'Draw a digit';
} catch (error) {
  statusEl.textContent = 'Could not load the model';
  throw error;
}

let drawing = false;
let last = null;
let pending = false;

clear();
pad.addEventListener('pointerdown', (event) => {
  drawing = true;
  pad.setPointerCapture(event.pointerId);
  last = point(event);
  strokeTo(last);
});
pad.addEventListener('pointermove', (event) => {
  if (drawing) strokeTo(point(event));
});
for (const type of ['pointerup', 'pointercancel']) {
  pad.addEventListener(type, () => { drawing = false; });
}
document.getElementById('clear').addEventListener('click', clear);

function clear() {
  padCtx.fillStyle = '#000';
  padCtx.fillRect(0, 0, pad.width, pad.height);
  padCtx.strokeStyle = '#fff';
  padCtx.lineWidth = 22;
  padCtx.lineCap = 'round';
  padCtx.lineJoin = 'round';
  seenCtx.clearRect(0, 0, 28, 28);
  guessEl.textContent = '?';
  confidenceEl.textContent = '';
  for (const row of rows) {
    row.li.classList.remove('best');
    row.fill.style.width = '0%';
    row.percent.textContent = '0.0%';
  }
}

function point(event) {
  const rect = pad.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (pad.width / rect.width),
    y: (event.clientY - rect.top) * (pad.height / rect.height),
  };
}

function strokeTo(next) {
  padCtx.beginPath();
  padCtx.moveTo(last.x, last.y);
  padCtx.lineTo(next.x, next.y);
  padCtx.stroke();
  last = next;
  if (!pending) {
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      predict();
    });
  }
}

function predict() {
  const pixels = toMnistPixels(padCtx.getImageData(0, 0, pad.width, pad.height).data, pad.width, pad.height);
  if (!pixels) return;
  drawSeen(pixels);
  smartdnn.HEAPF32.set(pixels, inputPtr / 4);
  const offset = smartdnn._predict(inputPtr) / 4;
  const probabilities = smartdnn.HEAPF32.slice(offset, offset + 10);
  const best = probabilities.indexOf(Math.max(...probabilities));
  guessEl.textContent = String(best);
  confidenceEl.textContent = `${(probabilities[best] * 100).toFixed(1)}% sure`;
  rows.forEach((row, digit) => {
    const percent = probabilities[digit] * 100;
    row.fill.style.width = `${percent}%`;
    row.percent.textContent = `${percent.toFixed(1)}%`;
    row.li.classList.toggle('best', digit === best);
  });
}

function drawSeen(pixels) {
  const image = seenCtx.createImageData(28, 28);
  pixels.forEach((value, i) => {
    const shade = Math.round(value * 255);
    image.data.set([shade, shade, shade, 255], i * 4);
  });
  seenCtx.putImageData(image, 0, 0);
}
