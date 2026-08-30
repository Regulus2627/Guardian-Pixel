import { ImageQualityMetrics } from '../types';

/**
 * Computes real Mean Squared Error (MSE) between two ImageData buffers
 */
export function computeMSE(data1: Uint8ClampedArray, data2: Uint8ClampedArray): {
  overall: number;
  r: number;
  g: number;
  b: number;
} {
  const len = Math.min(data1.length, data2.length);
  let sumR = 0, sumG = 0, sumB = 0;
  let pixelCount = 0;

  for (let i = 0; i < len; i += 4) {
    const diffR = data1[i] - data2[i];
    const diffG = data1[i + 1] - data2[i + 1];
    const diffB = data1[i + 2] - data2[i + 2];

    sumR += diffR * diffR;
    sumG += diffG * diffG;
    sumB += diffB * diffB;
    pixelCount++;
  }

  if (pixelCount === 0) return { overall: 0, r: 0, g: 0, b: 0 };

  const mseR = sumR / pixelCount;
  const mseG = sumG / pixelCount;
  const mseB = sumB / pixelCount;
  const overall = (mseR + mseG + mseB) / 3;

  return {
    overall: Number(overall.toFixed(4)),
    r: Number(mseR.toFixed(4)),
    g: Number(mseG.toFixed(4)),
    b: Number(mseB.toFixed(4)),
  };
}

/**
 * Computes real Peak Signal-to-Noise Ratio (PSNR) in dB from MSE
 */
export function computePSNR(mse: number, maxVal = 255): number {
  if (mse <= 1e-10) return 99.99; // Practically infinite / identical
  const psnr = 10 * Math.log10((maxVal * maxVal) / mse);
  return Number(Math.min(99.99, Math.max(0, psnr)).toFixed(2));
}

/**
 * Computes real Structural Similarity Index (SSIM) over 8x8 sliding blocks
 */
export function computeSSIM(
  data1: Uint8ClampedArray,
  data2: Uint8ClampedArray,
  width: number,
  height: number
): number {
  const C1 = 6.5025;  // (0.01 * 255)^2
  const C2 = 58.5225; // (0.03 * 255)^2

  const blockSize = 8;
  let totalSSIM = 0;
  let blockCount = 0;

  for (let y = 0; y <= height - blockSize; y += blockSize) {
    for (let x = 0; x <= width - blockSize; x += blockSize) {
      let sum1 = 0, sum2 = 0;
      let sumSq1 = 0, sumSq2 = 0, sumCross = 0;
      const N = blockSize * blockSize;

      for (let by = 0; by < blockSize; by++) {
        for (let bx = 0; bx < blockSize; bx++) {
          const idx = ((y + by) * width + (x + bx)) * 4;
          // Grayscale luminance conversion: 0.299R + 0.587G + 0.114B
          const p1 = 0.299 * data1[idx] + 0.587 * data1[idx + 1] + 0.114 * data1[idx + 2];
          const p2 = 0.299 * data2[idx] + 0.587 * data2[idx + 1] + 0.114 * data2[idx + 2];

          sum1 += p1;
          sum2 += p2;
          sumSq1 += p1 * p1;
          sumSq2 += p2 * p2;
          sumCross += p1 * p2;
        }
      }

      const mean1 = sum1 / N;
      const mean2 = sum2 / N;
      const var1 = (sumSq1 / N) - (mean1 * mean1);
      const var2 = (sumSq2 / N) - (mean2 * mean2);
      const covar = (sumCross / N) - (mean1 * mean2);

      const numerator = (2 * mean1 * mean2 + C1) * (2 * covar + C2);
      const denominator = (mean1 * mean1 + mean2 * mean2 + C1) * (var1 + var2 + C2);

      if (denominator > 0) {
        totalSSIM += numerator / denominator;
        blockCount++;
      }
    }
  }

  if (blockCount === 0) return 1.0;
  return Number(Math.max(0, Math.min(1.0, totalSSIM / blockCount)).toFixed(4));
}

/**
 * Computes complete real comparison metrics between cover image and stego image
 */
export function computeImageMetrics(
  coverData: ImageData,
  stegoData: ImageData,
  payloadBytes: number
): ImageQualityMetrics {
  const width = Math.min(coverData.width, stegoData.width);
  const height = Math.min(coverData.height, stegoData.height);

  const mse = computeMSE(coverData.data, stegoData.data);
  const psnrOverall = computePSNR(mse.overall);
  const psnrR = computePSNR(mse.r);
  const psnrG = computePSNR(mse.g);
  const psnrB = computePSNR(mse.b);

  const ssim = computeSSIM(coverData.data, stegoData.data, width, height);

  const totalPixels = width * height;
  const bpp = Number(((payloadBytes * 8) / (totalPixels * 3)).toFixed(4));

  const safeCap = (totalPixels * 3 * 0.25) / 8;
  const capacityUtilizationPct = Number(((payloadBytes / safeCap) * 100).toFixed(1));

  return {
    mse: mse.overall,
    psnr: psnrOverall,
    ssim,
    bpp,
    channels: {
      r: { mse: mse.r, psnr: psnrR },
      g: { mse: mse.g, psnr: psnrG },
      b: { mse: mse.b, psnr: psnrB },
    },
    capacityUtilizationPct,
    isSimulatedDetection: true,
  };
}

/**
 * Generates a real canvas difference map between two images
 */
export function renderDifferenceCanvas(
  canvas: HTMLCanvasElement,
  data1: ImageData,
  data2: ImageData,
  options: {
    amplification?: number;
    threshold?: number;
    overlayOpacity?: number;
    mode?: 'diff' | 'heatmap' | 'overlay';
  } = {}
) {
  const { amplification = 10, threshold = 0, overlayOpacity = 0.5, mode = 'diff' } = options;
  const width = Math.min(data1.width, data2.width);
  const height = Math.min(data1.height, data2.height);

  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const outData = ctx.createImageData(width, height);
  const d1 = data1.data;
  const d2 = data2.data;
  const out = outData.data;

  for (let i = 0; i < width * height * 4; i += 4) {
    const diffR = Math.abs(d1[i] - d2[i]);
    const diffG = Math.abs(d1[i + 1] - d2[i + 1]);
    const diffB = Math.abs(d1[i + 2] - d2[i + 2]);
    const maxDiff = Math.max(diffR, diffG, diffB);

    if (maxDiff < threshold) {
      if (mode === 'overlay') {
        out[i] = d1[i];
        out[i + 1] = d1[i + 1];
        out[i + 2] = d1[i + 2];
        out[i + 3] = 255;
      } else {
        out[i] = 0;
        out[i + 1] = 0;
        out[i + 2] = 0;
        out[i + 3] = 255;
      }
      continue;
    }

    const ampVal = Math.min(255, maxDiff * amplification);

    if (mode === 'heatmap') {
      // Magma/Turbo color mapping: blue -> green -> yellow -> red
      const norm = ampVal / 255;
      out[i] = Math.min(255, Math.floor(norm * 255 * 1.5));
      out[i + 1] = Math.min(255, Math.floor(Math.sin(norm * Math.PI) * 255));
      out[i + 2] = Math.min(255, Math.floor((1 - norm) * 255));
      out[i + 3] = 255;
    } else if (mode === 'overlay') {
      // Blend red error highlight onto original image
      const alpha = overlayOpacity;
      out[i] = Math.min(255, Math.floor((1 - alpha) * d1[i] + alpha * 255));
      out[i + 1] = Math.floor((1 - alpha) * d1[i + 1]);
      out[i + 2] = Math.floor((1 - alpha) * d1[i + 2]);
      out[i + 3] = 255;
    } else {
      // Standard grayscale or amplified RGB diff
      out[i] = Math.min(255, diffR * amplification);
      out[i + 1] = Math.min(255, diffG * amplification);
      out[i + 2] = Math.min(255, diffB * amplification);
      out[i + 3] = 255;
    }
  }

  ctx.putImageData(outData, 0, 0);
}
