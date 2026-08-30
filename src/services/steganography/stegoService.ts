import { apiClient } from '../apiClient/client';
import { StegoResult, ExtractionResult } from '../../types';

export interface EmbedOptions {
  coverDataUrl: string;
  payloadKind: 'text' | 'file' | 'image';
  rawPayload: string;
  payloadFilename?: string;
  payloadMimeType?: string;
  passphrase?: string;
  method?: string;
  coverMeta?: { width: number; height: number };
}

export interface ExtractOptions {
  stegoImageDataUrl: string;
  passphrase?: string;
}

/**
 * High-level Steganography Service
 * Dispatches to apiClient endpoints
 */
export async function embedPayload(options: EmbedOptions): Promise<StegoResult> {
  const res = await apiClient.post<StegoResult>('/embed', options);
  if (res.error) {
    throw new Error(res.error.message || 'Embedding failed');
  }
  if (!res.data) {
    throw new Error('No data received from embedding service');
  }
  return res.data;
}

export async function extractPayload(options: ExtractOptions): Promise<ExtractionResult> {
  const res = await apiClient.post<ExtractionResult>('/extract', options);
  if (res.error) {
    const err = new Error(res.error.message || 'Extraction failed');
    (err as any).code = res.error.code;
    throw err;
  }
  if (!res.data) {
    throw new Error('No data received from extraction service');
  }
  return res.data;
}

export async function assessDetectionRisk(bpp: number, textureScore: number) {
  const res = await apiClient.post<{ detectionScore: number; classifier: string; riskLevel: 'Low' | 'Moderate' | 'High'; isSimulated: boolean }>(
    '/metrics/detect',
    { bpp, textureScore }
  );
  if (res.error || !res.data) {
    return {
      detectionScore: Number((bpp * 1.2).toFixed(3)),
      classifier: 'SRM-39M Ensemble',
      riskLevel: 'Moderate' as const,
      isSimulated: true,
    };
  }
  return res.data;
}
