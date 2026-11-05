// Turns a white-on-black RGBA drawing into 784 MNIST-style values: bounding box scaled into 20 x 20, centred by mass in 28 x 28; null if blank.
export function toMnistPixels(rgba, width, height) {
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
