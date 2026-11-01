import createSmartDNN from './smartdnn.js';

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
  const pixels = toMnistPixels();
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

// Mirrors MNIST: the digit's bounding box is scaled to fit 20 x 20, then centred by mass in 28 x 28.
function toMnistPixels() {
  const { width, height } = pad;
  const rgba = padCtx.getImageData(0, 0, width, height).data;
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (rgba[(y * width + x) * 4] > 0) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }
  }
  if (maxX < 0) return null;

  const scale = 20 / Math.max(maxX - minX + 1, maxY - minY + 1);
  const w = Math.max(1, Math.round((maxX - minX + 1) * scale));
  const h = Math.max(1, Math.round((maxY - minY + 1) * scale));
  const small = new Float32Array(w * h);
  let mass = 0, massX = 0, massY = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const value = areaAverage(rgba, width, height, minX + x / scale, minY + y / scale, minX + (x + 1) / scale, minY + (y + 1) / scale);
      small[y * w + x] = value;
      mass += value;
      massX += value * x;
      massY += value * y;
    }
  }

  const dx = Math.min(Math.max(Math.round(13.5 - massX / mass), 0), 28 - w);
  const dy = Math.min(Math.max(Math.round(13.5 - massY / mass), 0), 28 - h);
  const pixels = new Float32Array(784);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) pixels[(y + dy) * 28 + x + dx] = small[y * w + x];
  }
  return pixels;
}

// Averages the red channel over a fractional source rectangle so the downscaled strokes stay anti-aliased.
function areaAverage(rgba, width, height, x0, y0, x1, y1) {
  let sum = 0, area = 0;
  for (let y = Math.floor(y0); y < Math.min(Math.ceil(y1), height); y++) {
    const wy = Math.min(y + 1, y1) - Math.max(y, y0);
    for (let x = Math.floor(x0); x < Math.min(Math.ceil(x1), width); x++) {
      const wx = Math.min(x + 1, x1) - Math.max(x, x0);
      sum += wx * wy * rgba[(y * width + x) * 4];
      area += wx * wy;
    }
  }
  return area > 0 ? sum / area / 255 : 0;
}

function drawSeen(pixels) {
  const image = seenCtx.createImageData(28, 28);
  pixels.forEach((value, i) => {
    const shade = Math.round(value * 255);
    image.data.set([shade, shade, shade, 255], i * 4);
  });
  seenCtx.putImageData(image, 0, 0);
}
