import { CapacityResult, CompatibilityResult, CoverImageMeta } from '../types';

export const MIN_SUPPORTED_DIMENSION = 64;
export const SUPPORTED_FORMATS = ['PNG', 'JPEG', 'JPG', 'WEBP', 'BMP'];

/**
 * Computes real capacity arithmetic for a cover image and current payload
 */
export function calculateCapacity(
  meta: CoverImageMeta | null,
  currentPayloadBytes: number,
  textureScore = 0.5
): CapacityResult {
  if (!meta || meta.width <= 0 || meta.height <= 0) {
    return {
      maxTheoreticalBytes: 0,
      recommendedSafeBytes: 0,
      currentPayloadBytes,
      utilizationPct: 0,
    };
  }

  const totalPixels = meta.width * meta.height;
  const channels = meta.channels || 3;

  // Max theoretical 1-bit LSB: 1 bit per channel per pixel
  const maxTheoreticalBytes = Math.floor((totalPixels * channels) / 8);

  // Recommended safe capacity: safe bpp between 0.10 and 0.35 based on cover texture richness
  const baseSafeBpp = 0.15 + (Math.max(0, Math.min(1, textureScore)) * 0.20);
  const recommendedSafeBytes = Math.max(32, Math.floor((totalPixels * channels * baseSafeBpp) / 8));

  const utilizationPct = recommendedSafeBytes > 0
    ? Number(((currentPayloadBytes / recommendedSafeBytes) * 100).toFixed(1))
    : 0;

  return {
    maxTheoreticalBytes,
    recommendedSafeBytes,
    currentPayloadBytes,
    utilizationPct,
  };
}

/**
 * Evaluates real compatibility classification based on texture and headroom ratio
 */
export function evaluateCompatibility(
  meta: CoverImageMeta | null,
  payloadBytes: number,
  textureScore = 0.5
): CompatibilityResult {
  if (!meta) {
    return {
      textureScore: 0,
      headroomRatio: 0,
      classification: 'Unsupported',
      recommended: false,
      reason: 'No cover image loaded.',
    };
  }

  const isSupportedFormat = SUPPORTED_FORMATS.includes(meta.fileType.toUpperCase());
  const hasMinimumResolution = meta.width >= MIN_SUPPORTED_DIMENSION && meta.height >= MIN_SUPPORTED_DIMENSION;

  if (!isSupportedFormat || !hasMinimumResolution) {
    let reason = 'Unsupported cover image.';
    if (!isSupportedFormat) {
      reason = `Format ${meta.fileType} is not supported for lossless steganographic carrier. Use PNG, WEBP, or BMP.`;
    } else if (!hasMinimumResolution) {
      reason = `Dimensions (${meta.width}x${meta.height}) are below minimum required ${MIN_SUPPORTED_DIMENSION}x${MIN_SUPPORTED_DIMENSION}px.`;
    }

    return {
      textureScore: Number(textureScore.toFixed(3)),
      headroomRatio: 0,
      classification: 'Unsupported',
      recommended: false,
      reason,
      details: {
        meanVariance: 0,
        meanEntropy: 0,
        normalizedTexture: textureScore,
        bppTarget: 0,
        isSupportedFormat,
        hasMinimumResolution,
      },
    };
  }

  const capacity = calculateCapacity(meta, payloadBytes, textureScore);
  const safeCap = Math.max(1, capacity.recommendedSafeBytes);
  const activePayload = Math.max(1, payloadBytes);
  const headroomRatio = Number((safeCap / activePayload).toFixed(2));

  let classification: CompatibilityResult['classification'] = 'Moderate';
  let recommended = true;
  let reason = '';

  // Band rules from requirements:
  // Excellent: headroom >= 4x and texture in top tercile (>= 0.67)
  // Good: headroom >= 2x and texture >= median (>= 0.40)
  // Moderate: headroom >= 1x regardless of texture
  // Poor: headroom < 1x but format supported
  // Unsupported: bad format or below minimum dimensions

  if (headroomRatio >= 4.0 && textureScore >= 0.67) {
    classification = 'Excellent';
    recommended = true;
    reason = `Outstanding cover fidelity: ${headroomRatio}x safe headroom with rich spatial texture (score ${textureScore.toFixed(2)}) providing optimal steganalysis resistance.`;
  } else if (headroomRatio >= 2.0 && textureScore >= 0.40) {
    classification = 'Good';
    recommended = true;
    reason = `Solid cover compatibility: ${headroomRatio}x safe headroom with sufficient texture variance to mask LSB perturbations.`;
  } else if (headroomRatio >= 1.0) {
    classification = 'Moderate';
    recommended = true;
    reason = `Acceptable capacity fit (${headroomRatio}x headroom), but lower texture variance or higher payload density may increase statistical detectability.`;
  } else {
    classification = 'Poor';
    recommended = false;
    reason = `Payload exceeds recommended safe capacity (${headroomRatio}x headroom). Significant risk of perceptual artifacts and steganalytic detection.`;
  }

  return {
    textureScore: Number(textureScore.toFixed(3)),
    headroomRatio,
    classification,
    recommended,
    reason,
    details: {
      meanVariance: 0,
      meanEntropy: 0,
      normalizedTexture: textureScore,
      bppTarget: Number(((payloadBytes * 8) / (meta.width * meta.height * 3)).toFixed(4)),
      isSupportedFormat,
      hasMinimumResolution,
    },
  };
}
