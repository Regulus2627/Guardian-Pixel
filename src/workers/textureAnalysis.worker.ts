// Web Worker for Real Texture, Variance, Entropy, Sobel Edge, Fusion & Binary Segmentation Analysis

interface WorkerInput {
  pixels: Uint8ClampedArray;
  width: number;
  height: number;
}

interface WorkerOutput {
  textureScore: number;
  meanVariance: number;
  meanEntropy: number;
  edgeDensity: number;
  width: number;
  height: number;
  edgeMapPixels: Uint8ClampedArray;
  entropyMapPixels: Uint8ClampedArray;
  varianceMapPixels: Uint8ClampedArray;
  fusionMapPixels: Uint8ClampedArray;
  binaryMapPixels: Uint8ClampedArray;
}

self.onmessage = function (e: MessageEvent<WorkerInput>) {
  const { pixels, width, height } = e.data;
  const numPixels = width * height;

  // 1. Convert to grayscale luminance array
  const gray = new Float32Array(numPixels);
  for (let i = 0; i < numPixels; i++) {
    const idx = i * 4;
    gray[i] = 0.299 * pixels[idx] + 0.587 * pixels[idx + 1] + 0.114 * pixels[idx + 2];
  }

  // 2. Sobel Edge Detection
  const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  const edgeMagnitudes = new Float32Array(numPixels);
  let maxEdge = 0;
  let edgeSum = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0;
      let gy = 0;
      let k = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const val = gray[(y + dy) * width + (x + dx)];
          gx += val * sobelX[k];
          gy += val * sobelY[k];
          k++;
        }
      }
      const mag = Math.sqrt(gx * gx + gy * gy);
      const idx = y * width + x;
      edgeMagnitudes[idx] = mag;
      if (mag > maxEdge) maxEdge = mag;
      edgeSum += mag;
    }
  }

  // 3. Local Spatial Variance (5x5 window)
  const variances = new Float32Array(numPixels);
  let maxVar = 0;
  let varSum = 0;
  const vRadius = 2; // 5x5

  for (let y = vRadius; y < height - vRadius; y += 1) {
    for (let x = vRadius; x < width - vRadius; x += 1) {
      let sum = 0;
      let sumSq = 0;
      let count = 0;

      for (let dy = -vRadius; dy <= vRadius; dy++) {
        for (let dx = -vRadius; dx <= vRadius; dx++) {
          const val = gray[(y + dy) * width + (x + dx)];
          sum += val;
          sumSq += val * val;
          count++;
        }
      }

      const mean = sum / count;
      const v = (sumSq / count) - (mean * mean);
      const idx = y * width + x;
      variances[idx] = Math.max(0, v);
      if (v > maxVar) maxVar = v;
      varSum += v;
    }
  }

  // 4. Local Shannon Entropy (7x7 window, histogram of 16 bins for performance)
  const entropies = new Float32Array(numPixels);
  let maxEntropy = 0;
  let entropySum = 0;
  const eRadius = 3; // 7x7
  const bins = new Uint16Array(16);

  for (let y = eRadius; y < height - eRadius; y += 2) {
    for (let x = eRadius; x < width - eRadius; x += 2) {
      bins.fill(0);
      let count = 0;

      for (let dy = -eRadius; dy <= eRadius; dy++) {
        for (let dx = -eRadius; dx <= eRadius; dx++) {
          const val = gray[(y + dy) * width + (x + dx)];
          const binIdx = Math.min(15, Math.floor(val / 16));
          bins[binIdx]++;
          count++;
        }
      }

      let ent = 0;
      for (let b = 0; b < 16; b++) {
        if (bins[b] > 0) {
          const p = bins[b] / count;
          ent -= p * Math.log2(p);
        }
      }

      // Fill 2x2 block
      for (let dy = 0; dy < 2 && y + dy < height; dy++) {
        for (let dx = 0; dx < 2 && x + dx < width; dx++) {
          const idx = (y + dy) * width + (x + dx);
          entropies[idx] = ent;
        }
      }

      if (ent > maxEntropy) maxEntropy = ent;
      entropySum += ent;
    }
  }

  // 5. Output Buffers (RGBA)
  const edgeMapPixels = new Uint8ClampedArray(numPixels * 4);
  const entropyMapPixels = new Uint8ClampedArray(numPixels * 4);
  const varianceMapPixels = new Uint8ClampedArray(numPixels * 4);
  const fusionMapPixels = new Uint8ClampedArray(numPixels * 4);
  const binaryMapPixels = new Uint8ClampedArray(numPixels * 4);

  const safeMaxEdge = maxEdge > 0 ? maxEdge : 1;
  const safeMaxVar = maxVar > 0 ? maxVar : 1;
  const safeMaxEntropy = maxEntropy > 0 ? maxEntropy : 1;

  // Multi-feature fusion weights: 35% Edge, 35% Variance, 30% Entropy
  let totalFusionScore = 0;

  for (let i = 0; i < numPixels; i++) {
    const idx = i * 4;

    const normEdge = Math.min(1, edgeMagnitudes[i] / safeMaxEdge);
    const normVar = Math.min(1, Math.sqrt(variances[i]) / Math.sqrt(safeMaxVar));
    const normEnt = Math.min(1, entropies[i] / safeMaxEntropy);

    // Sobel (electric cyan/blue gradient)
    const eVal = Math.floor(normEdge * 255);
    edgeMapPixels[idx] = Math.floor(eVal * 0.2);
    edgeMapPixels[idx + 1] = Math.floor(eVal * 0.85);
    edgeMapPixels[idx + 2] = eVal;
    edgeMapPixels[idx + 3] = 255;

    // Variance (amber/gold heatmap)
    const vVal = Math.floor(normVar * 255);
    varianceMapPixels[idx] = vVal;
    varianceMapPixels[idx + 1] = Math.floor(vVal * 0.75);
    varianceMapPixels[idx + 2] = Math.floor(vVal * 0.2);
    varianceMapPixels[idx + 3] = 255;

    // Entropy (emerald/green heatmap)
    const entVal = Math.floor(normEnt * 255);
    entropyMapPixels[idx] = Math.floor(entVal * 0.1);
    entropyMapPixels[idx + 1] = entVal;
    entropyMapPixels[idx + 2] = Math.floor(entVal * 0.45);
    entropyMapPixels[idx + 3] = 255;

    // Fusion: combined texture mask
    const fusion = 0.35 * normEdge + 0.35 * normVar + 0.30 * normEnt;
    totalFusionScore += fusion;

    // Turbo-like pseudo color for fusion
    fusionMapPixels[idx] = Math.min(255, Math.floor(fusion * 1.8 * 255));
    fusionMapPixels[idx + 1] = Math.floor(Math.sin(fusion * Math.PI) * 255);
    fusionMapPixels[idx + 2] = Math.max(0, Math.floor((1 - fusion) * 255));
    fusionMapPixels[idx + 3] = 255;

    // Binary Segmentation (high capacity carrier zones: top 45% fusion threshold)
    const isCarrierZone = fusion >= 0.32;
    binaryMapPixels[idx] = isCarrierZone ? 56 : 15;     // Slate background / Cyan active zone
    binaryMapPixels[idx + 1] = isCarrierZone ? 189 : 23;
    binaryMapPixels[idx + 2] = isCarrierZone ? 248 : 42;
    binaryMapPixels[idx + 3] = 255;
  }

  // Normalized overall metrics
  const meanVar = varSum / numPixels;
  const meanEnt = entropySum / (numPixels / 4);
  const textureScore = Number(Math.min(1.0, Math.max(0.05, (totalFusionScore / numPixels) * 1.6)).toFixed(3));
  const edgeDensity = Number((edgeSum / (numPixels * 255)).toFixed(3));

  const result: WorkerOutput = {
    textureScore,
    meanVariance: Number(meanVar.toFixed(2)),
    meanEntropy: Number(meanEnt.toFixed(2)),
    edgeDensity,
    width,
    height,
    edgeMapPixels,
    entropyMapPixels,
    varianceMapPixels,
    fusionMapPixels,
    binaryMapPixels,
  };

  (self as any).postMessage(result, [
    edgeMapPixels.buffer,
    entropyMapPixels.buffer,
    varianceMapPixels.buffer,
    fusionMapPixels.buffer,
    binaryMapPixels.buffer,
  ]);
};
