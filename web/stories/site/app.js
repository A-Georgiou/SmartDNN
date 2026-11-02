import createStoryGpt from './storygpt.js';

const MAX_TOKENS = 600;
const TOP_K = 40;

const promptInput = document.getElementById('prompt');
const temperatureInput = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperature-value');
const button = document.getElementById('generate');
const output = document.getElementById('output');
const statusEl = document.getElementById('status');

let story;
let running = false;

temperatureInput.addEventListener('input', () => {
  temperatureValue.textContent = Number(temperatureInput.value).toFixed(2);
});
for (const example of document.querySelectorAll('.example')) {
  example.addEventListener('click', () => {
    promptInput.value = example.textContent;
    if (!running && story) start();
  });
}
button.addEventListener('click', () => (running ? stop() : start()));

try {
  story = await createStoryGpt();
  story._story_load();
  statusEl.textContent = 'Ready';
  button.disabled = false;
} catch (error) {
  statusEl.textContent = 'Could not load the model';
  console.error(error);
}

function start() {
  const prompt = promptInput.value.trim();
  const decoder = new TextDecoder();
  const startedAt = performance.now();
  let text = prompt;
  let written = 0;
  story.ccall('story_begin', null, ['string', 'number'], [prompt, Math.floor(Math.random() * 2 ** 32)]);
  running = true;
  button.textContent = 'Stop';

  // Writes for about one frame at a time so the page stays responsive while the story streams in.
  const step = () => {
    if (!running) return;
    const frameEnd = performance.now() + 25;
    let ended = false;
    while (written < MAX_TOKENS && performance.now() < frameEnd) {
      const length = story._story_next(Number(temperatureInput.value), TOP_K);
      if (length < 0) {
        ended = true;
        break;
      }
      const pointer = story._story_piece();
      text += decoder.decode(story.HEAPU8.subarray(pointer, pointer + length), { stream: true });
      written++;
    }
    output.textContent = text;
    output.scrollTop = output.scrollHeight;
    const perSecond = written / ((performance.now() - startedAt) / 1000);
    statusEl.textContent = `${written} tokens, ${perSecond.toFixed(0)} per second${ended ? ', the end' : ''}`;
    if (ended || written >= MAX_TOKENS) stop();
    else requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function stop() {
  running = false;
  button.textContent = 'Write';
}
