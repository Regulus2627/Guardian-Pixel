import { apiClient } from '../apiClient/client';

export type CompatibilityClassification =
  | 'Excellent'
  | 'Good'
  | 'Moderate'
  | 'Poor';

export interface CompatibilityRequest {
  coverDataUrl: string;
  payloadKind: 'text' | 'file' | 'image';
  rawPayload: string;
  payloadFilename?: string;
  payloadMimeType?: string;
}

export interface RealCompatibilityResult {
  textureScore: number;
  headroomRatio: number;
  classification: CompatibilityClassification;
  recommended: boolean;
  compatible: boolean;
  reason: string;
  isProvisional: boolean;
  isSimulated: boolean;

  cover: {
    category: 'Smooth' | 'Mixed' | 'Textured';
    width: number;
    height: number;
    totalPixels: number;
  };

  payload: {
    kind: 'text' | 'file' | 'image';
    originalBytes: number;
    envelopeBytes: number;
    encryptedPacketBits: number;
  };

  capacity: {
    bootstrapBits: number;
    metadataBits: number;
    totalRequiredBits: number;
    requiredWithSafety: number;
    maximumPositions: number;
    remainingPositions: number;
  };

  locationMap: {
    selectedBlocks: number;
    totalBlocks: number;
    selectedPercentage: number;
    encoding: string;
    encodedBytes: number;
  };

  details?: {
    bppTarget?: number;
    isSupportedFormat?: boolean;
    hasMinimumResolution?: boolean;
  };
}

export async function assessRealCompatibility(
  request: CompatibilityRequest
): Promise<RealCompatibilityResult> {
  const response = await apiClient.post<RealCompatibilityResult>(
    '/compatibility',
    request
  );

  if (response.error) {
    throw new Error(response.error.message || 'Compatibility analysis failed.');
  }

  if (!response.data) {
    throw new Error('No compatibility result was returned by the backend.');
  }

  if (response.data.isSimulated) {
    throw new Error(
      'A simulated compatibility response was received. Start Flask and disable mock mode.'
    );
  }

  return response.data;
}
