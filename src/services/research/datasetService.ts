import { ExperimentRow } from '../../types';

// Linear Congruential Generator for reproducible deterministic seeded pseudo-random values
class SeededRandom {
  private seed: number;

  constructor(seed = 42891) {
    this.seed = seed % 2147483647;
    if (this.seed <= 0) this.seed += 2147483646;
  }

  next(): number {
    this.seed = (this.seed * 16807) % 2147483647;
    return (this.seed - 1) / 2147483646;
  }

  range(min: number, max: number): number {
    return min + this.next() * (max - min);
  }

  choice<T>(arr: readonly T[] | T[]): T {
    return arr[Math.floor(this.next() * arr.length)];
  }
}

export const METHODS = [
  'LSB-Sequential',
  'Adaptive-Edge-LSB',
  'DCT-Frequency-Domain',
  'StegEx-ExactNet',
  'DenseAutoencoder-Approx',
] as const;

export const CATEGORIES = [
  'Natural',
  'Portrait',
  'Urban',
  'Texture-rich',
  'Synthetic',
  'Low-texture',
] as const;

export const RESOLUTIONS = [
  '512x512',
  '1024x1024',
  '1920x1080',
  '2048x2048',
] as const;

/**
 * Generates deterministic 120+ Experiment benchmark rows
 */
export function generateDeterministicDataset(seed = 94820): ExperimentRow[] {
  const rng = new SeededRandom(seed);
  const rows: ExperimentRow[] = [];

  let idCounter = 1;

  for (const category of CATEGORIES) {
    // Base capacity multiplier by category texture profile
    let categoryTextureMod = 1.0;
    if (category === 'Texture-rich') categoryTextureMod = 1.6;
    else if (category === 'Urban') categoryTextureMod = 1.25;
    else if (category === 'Natural') categoryTextureMod = 1.1;
    else if (category === 'Portrait') categoryTextureMod = 0.85;
    else if (category === 'Synthetic') categoryTextureMod = 0.75;
    else if (category === 'Low-texture') categoryTextureMod = 0.5;

    for (const method of METHODS) {
      // 4 distinct bpp test points per method/category combination
      const bppPoints = [0.05, 0.15, 0.35, 0.75];

      for (const targetBpp of bppPoints) {
        const resolution = rng.choice(RESOLUTIONS);
        const [w, h] = resolution.split('x').map(Number);
        const totalPixels = w * h;

        // Add small deterministic jitter
        const bpp = Number((targetBpp + rng.range(-0.015, 0.015)).toFixed(3));
        const payloadBytes = Math.floor((totalPixels * 3 * bpp) / 8);
        const safeCapacityBytes = Math.floor((totalPixels * 3 * 0.25 * categoryTextureMod) / 8);

        // Compute realistic physical metrics correlated to method & bpp
        let basePsnr = 52.0;
        let baseSsim = 0.992;
        let baseMse = 0.45;
        let baseDetection = 0.10;
        let baseTime = 180;

        if (method === 'LSB-Sequential') {
          basePsnr = 55.0 - (bpp * 18.0) + rng.range(-0.8, 0.8);
          baseSsim = 0.996 - (bpp * 0.045) + rng.range(-0.003, 0.003);
          baseMse = 0.35 + (bpp * 3.8);
          baseDetection = 0.08 + (bpp * 0.85) / categoryTextureMod;
          baseTime = 95 + (payloadBytes / 1024) * 0.15;
        } else if (method === 'Adaptive-Edge-LSB') {
          basePsnr = 53.5 - (bpp * 14.0) + (categoryTextureMod * 1.5) + rng.range(-0.6, 0.6);
          baseSsim = 0.994 - (bpp * 0.03) + rng.range(-0.002, 0.002);
          baseMse = 0.42 + (bpp * 2.9);
          baseDetection = 0.05 + (bpp * 0.42) / categoryTextureMod;
          baseTime = 220 + (payloadBytes / 1024) * 0.4;
        } else if (method === 'DCT-Frequency-Domain') {
          basePsnr = 49.0 - (bpp * 15.0) + rng.range(-1.2, 1.2);
          baseSsim = 0.988 - (bpp * 0.05) + rng.range(-0.005, 0.005);
          baseMse = 0.85 + (bpp * 5.2);
          baseDetection = 0.12 + (bpp * 0.55) / categoryTextureMod;
          baseTime = 340 + (payloadBytes / 1024) * 0.8;
        } else if (method === 'StegEx-ExactNet') {
          basePsnr = 56.5 - (bpp * 10.5) + rng.range(-0.5, 0.5);
          baseSsim = 0.998 - (bpp * 0.015) + rng.range(-0.001, 0.001);
          baseMse = 0.28 + (bpp * 1.8);
          baseDetection = 0.03 + (bpp * 0.28) / categoryTextureMod;
          baseTime = 480 + (payloadBytes / 1024) * 0.6;
        } else if (method === 'DenseAutoencoder-Approx') {
          basePsnr = 44.0 - (bpp * 12.0) + rng.range(-1.5, 1.5);
          baseSsim = 0.965 - (bpp * 0.075) + rng.range(-0.008, 0.008);
          baseMse = 2.4 + (bpp * 8.5);
          baseDetection = 0.15 + (bpp * 0.48) / categoryTextureMod;
          baseTime = 650 + (payloadBytes / 1024) * 1.1;
        }

        const psnr = Number(Math.max(28, Math.min(65, basePsnr)).toFixed(2));
        const ssim = Number(Math.max(0.85, Math.min(0.9999, baseSsim)).toFixed(4));
        const mse = Number(Math.max(0.01, baseMse).toFixed(3));
        const detectionScore = Number(Math.max(0.01, Math.min(0.99, baseDetection)).toFixed(3));
        const execTimeMs = Math.floor(Math.max(40, baseTime + rng.range(-20, 20)));

        rows.push({
          id: `EXP-${idCounter.toString().padStart(4, '0')}`,
          method,
          category,
          resolution,
          payloadBytes,
          bpp,
          psnr,
          ssim,
          mse,
          detectionScore,
          safeCapacityBytes,
          execTimeMs,
          isDemoData: true,
        });

        idCounter++;
      }
    }
  }

  return rows;
}

export interface StatisticalSummary {
  count: number;
  meanPsnr: number;
  sdPsnr: number;
  ci95Psnr: [number, number];
  meanSsim: number;
  sdSsim: number;
  ci95Ssim: [number, number];
  meanMse: number;
  sdMse: number;
  meanBpp: number;
  meanDetection: number;
  meanExecTime: number;
}

/**
 * Computes mean, standard deviation, and 95% Confidence Interval for an array of numbers
 */
function calculateStats(values: number[]): { mean: number; sd: number; ci95: [number, number] } {
  if (values.length === 0) return { mean: 0, sd: 0, ci95: [0, 0] };
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  if (values.length === 1) return { mean, sd: 0, ci95: [mean, mean] };

  const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (values.length - 1);
  const sd = Math.sqrt(variance);
  const standardError = sd / Math.sqrt(values.length);
  // Z-score 1.96 for 95% CI
  const margin = 1.96 * standardError;

  return {
    mean: Number(mean.toFixed(3)),
    sd: Number(sd.toFixed(3)),
    ci95: [Number((mean - margin).toFixed(3)), Number((mean + margin).toFixed(3))],
  };
}

export function computeDatasetStatistics(rows: ExperimentRow[]): StatisticalSummary {
  if (rows.length === 0) {
    return {
      count: 0,
      meanPsnr: 0,
      sdPsnr: 0,
      ci95Psnr: [0, 0],
      meanSsim: 0,
      sdSsim: 0,
      ci95Ssim: [0, 0],
      meanMse: 0,
      sdMse: 0,
      meanBpp: 0,
      meanDetection: 0,
      meanExecTime: 0,
    };
  }

  const psnrStats = calculateStats(rows.map(r => r.psnr));
  const ssimStats = calculateStats(rows.map(r => r.ssim));
  const mseStats = calculateStats(rows.map(r => r.mse));

  const meanBpp = Number((rows.reduce((s, r) => s + r.bpp, 0) / rows.length).toFixed(3));
  const meanDetection = Number((rows.reduce((s, r) => s + r.detectionScore, 0) / rows.length).toFixed(3));
  const meanExecTime = Math.round(rows.reduce((s, r) => s + r.execTimeMs, 0) / rows.length);

  return {
    count: rows.length,
    meanPsnr: psnrStats.mean,
    sdPsnr: psnrStats.sd,
    ci95Psnr: psnrStats.ci95,
    meanSsim: ssimStats.mean,
    sdSsim: ssimStats.sd,
    ci95Ssim: ssimStats.ci95,
    meanMse: mseStats.mean,
    sdMse: mseStats.sd,
    meanBpp,
    meanDetection,
    meanExecTime,
  };
}
