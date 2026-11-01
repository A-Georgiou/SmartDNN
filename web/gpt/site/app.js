import createCharGpt from './chargpt.js';

const promptInput = document.getElementById('prompt');
const temperatureInput = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperature-value');
const lengthInput = document.getElementById('length');
const lengthValue = document.getElementById('length-value');
const button = document.getElementById('generate');
const output = document.getElementById('output');
const statusEl = document.getElementById('status');

let gpt;
let running = false;

temperatureInput.addEventListener('input', () => {
  temperatureValue.textContent = Number(temperatureInput.value).toFixed(2);
});
lengthInput.addEventListener('input', () => {
  lengthValue.textContent = lengthInput.value;
});
button.addEventListener('click', () => (running ? stop() : start()));

try {
  gpt = await createCharGpt();
  gpt._gpt_load();
  statusEl.textContent = 'Ready';
  button.disabled = false;
} catch (error) {
  statusEl.textContent = 'Could not load the model';
  console.error(error);
}

function start() {
  const total = Number(lengthInput.value);
  const startedAt = performance.now();
  let text = promptInput.value;
  let written = 0;
  gpt.ccall('gpt_reset', null, ['string', 'number'], [text, Math.floor(Math.random() * 2 ** 32)]);
  running = true;
  button.textContent = 'Stop';

  // Writes for about one frame at a time so the page stays responsive while the text streams in.
  const step = () => {
    if (!running) return;
    const frameEnd = performance.now() + 25;
    while (written < total && performance.now() < frameEnd) {
      text += String.fromCharCode(gpt._gpt_next(Number(temperatureInput.value)));
      written++;
    }
    output.textContent = text;
    output.scrollTop = output.scrollHeight;
    const perSecond = written / ((performance.now() - startedAt) / 1000);
    statusEl.textContent = `${written} characters, ${perSecond.toFixed(0)} per second`;
    if (written < total) requestAnimationFrame(step);
    else stop();
  };
  requestAnimationFrame(step);
}

function stop() {
  running = false;
  button.textContent = 'Generate';
}
