export interface CoverImageMeta {
  width: number;
  height: number;
  colorMode: string;
  fileType: string;
  fileSizeBytes: number;
  bitDepth?: number;
  channels: number;
}

export interface PayloadInfo {
  kind: 'text' | 'file' | 'image';
  rawBytes: number;
  compressedBytes?: number;
  encryptedBytes?: number;
  isSimulated: boolean;
  filename?: string;
  mimeType?: string;
  dataUrl?: string; // For image payloads
  textContent?: string; // For text payloads
}

export interface CapacityResult {
  maxTheoreticalBytes: number;
  recommendedSafeBytes: number;
  currentPayloadBytes: number;
  utilizationPct: number;
}

export interface CompatibilityResult {
  textureScore: number;
  headroomRatio: number;
  classification: 'Excellent' | 'Good' | 'Moderate' | 'Poor' | 'Unsupported';
  recommended: boolean;
  reason: string;
  details?: {
    meanVariance: number;
    meanEntropy: number;
    normalizedTexture: number;
    bppTarget: number;
    isSupportedFormat: boolean;
    hasMinimumResolution: boolean;
  };
}

export interface StegoResult {
  stegoImageDataUrl: string;
  embedTimeMs: number;
  method: string;
  bpp: number;
  isSimulated: boolean;
  payloadInfo: PayloadInfo;
  sessionToken?: string;
}

export interface ExtractionResult {
  kind: 'text' | 'file' | 'image';
  textContent?: string;
  fileDataUrl?: string;
  filename?: string;
  mimeType?: string;
  fileSizeBytes?: number;
  extractTimeMs: number;
  integrityVerified: boolean;
  method: string;
  isSimulated: boolean;
}

export interface ExperimentRow {
  id?: string;
  method: string;
  category: string;
  resolution: string;
  payloadBytes: number;
  bpp: number;
  psnr: number;
  ssim: number;
  mse: number;
  detectionScore: number;
  safeCapacityBytes: number;
  execTimeMs: number;
  isDemoData: true;
}

export interface ImageQualityMetrics {
  mse: number;
  psnr: number;
  ssim: number;
  bpp: number;
  channels: {
    r: { mse: number; psnr: number };
    g: { mse: number; psnr: number };
    b: { mse: number; psnr: number };
  };
  capacityUtilizationPct: number;
  embedTimeMs?: number;
  extractTimeMs?: number;
  detectionScore?: number;
  isSimulatedDetection: boolean;
}

export interface TextureAnalysisResult {
  textureScore: number;
  meanVariance: number;
  meanEntropy: number;
  edgeDensity: number;
  width: number;
  height: number;
  edgeMapDataUrl: string;
  entropyMapDataUrl: string;
  varianceMapDataUrl: string;
  fusionMapDataUrl: string;
  binaryMapDataUrl: string;
}

export interface SampleImageItem {
  id: string;
  title: string;
  category: 'Natural' | 'Portrait' | 'Urban' | 'Texture-rich' | 'Synthetic' | 'Low-texture';
  description: string;
  expectedTexture: 'High' | 'Medium' | 'Low';
  dataUrl: string;
  previewColor: string;
  dimensions: { width: number; height: number };
}
