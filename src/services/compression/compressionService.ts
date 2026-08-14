import { apiClient } from '../apiClient/client';

export interface CompressionEstimateResult {
  compressedBytes: number;
  compressionRatio: number;
  isSimulated: boolean;
}

export async function estimateCompressedSize(
  rawBytes: number,
  kind: 'text' | 'file' | 'image'
): Promise<CompressionEstimateResult> {
  const res = await apiClient.post<CompressionEstimateResult>('/compression/estimate', { rawBytes, kind });
  if (res.error || !res.data) {
    let ratio = 0.5;
    if (kind === 'text') ratio = 0.45;
    if (kind === 'file') ratio = 0.82;
    if (kind === 'image') ratio = 0.95;
    return {
      compressedBytes: Math.max(16, Math.floor(rawBytes * ratio)),
      compressionRatio: ratio,
      isSimulated: true,
    };
  }
  return res.data;
}
