import { apiClient } from '../apiClient/client';

export interface CryptoEstimateResult {
  encryptedBytes: number;
  overheadBytes: number;
  isSimulated: boolean;
}

export async function estimateEncryptedSize(rawBytes: number, algorithm = 'AES-256-GCM'): Promise<CryptoEstimateResult> {
  const res = await apiClient.post<CryptoEstimateResult>('/crypto/estimate', { rawBytes, algorithm });
  if (res.error || !res.data) {
    return {
      encryptedBytes: rawBytes + 32,
      overheadBytes: 32,
      isSimulated: true,
    };
  }
  return res.data;
}
