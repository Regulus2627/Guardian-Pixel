import { TextureAnalysisResult } from '../types';
import { extractImageData } from '../utils/imageProcessing';

/**
 * Converts raw pixel buffer back to PNG Data URL via offscreen/temporary canvas
 */
function pixelsToDataUrl(pixels: Uint8ClampedArray, width: number, height: number): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  const imgData = ctx.createImageData(width, height);
  imgData.data.set(pixels);
  ctx.putImageData(imgData, 0, 0);

  return canvas.toDataURL('image/png');
}

/**
 * Runs full real texture & edge analysis using a Web Worker
 */
export async function analyzeCoverTexture(coverSrc: string): Promise<TextureAnalysisResult> {
  const { imageData, width, height } = await extractImageData(coverSrc, 1200);

  return new Promise((resolve, reject) => {
    // Vite Web Worker syntax
    const worker = new Worker(
      new URL('./textureAnalysis.worker.ts', import.meta.url),
      { type: 'module' }
    );

    worker.onmessage = (e: MessageEvent) => {
      const data = e.data;

      const result: TextureAnalysisResult = {
        textureScore: data.textureScore,
        meanVariance: data.meanVariance,
        meanEntropy: data.meanEntropy,
        edgeDensity: data.edgeDensity,
        width: data.width,
        height: data.height,
        edgeMapDataUrl: pixelsToDataUrl(data.edgeMapPixels, data.width, data.height),
        entropyMapDataUrl: pixelsToDataUrl(data.entropyMapPixels, data.width, data.height),
        varianceMapDataUrl: pixelsToDataUrl(data.varianceMapPixels, data.width, data.height),
        fusionMapDataUrl: pixelsToDataUrl(data.fusionMapPixels, data.width, data.height),
        binaryMapDataUrl: pixelsToDataUrl(data.binaryMapPixels, data.width, data.height),
      };

      worker.terminate();
      resolve(result);
    };

    worker.onerror = (err) => {
      worker.terminate();
      reject(err);
    };

    // Transfer pixel buffer to worker for zero-copy high performance
    worker.postMessage(
      {
        pixels: imageData.data,
        width,
        height,
      }
    );
  });
}
