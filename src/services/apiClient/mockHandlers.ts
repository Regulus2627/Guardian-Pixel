import { StegoResult, ExtractionResult, PayloadInfo } from '../../types';

// In-memory session registry for deterministic mock extraction
interface SessionEmbedRecord {
  id: string;
  fingerprint: string;
  passphraseHash: string;
  payloadInfo: PayloadInfo;
  method: string;
  timestamp: number;
}

const sessionRegistry: SessionEmbedRecord[] = [];

// Simple deterministic hash for mock auth and fingerprinting
function simpleHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}

/**
 * Generates an image fingerprint from data URL sample
 */
function getImageFingerprint(dataUrl: string): string {
  if (!dataUrl) return 'empty';
  const prefix = dataUrl.slice(0, 100);
  const middle = dataUrl.slice(Math.floor(dataUrl.length / 2), Math.floor(dataUrl.length / 2) + 100);
  const suffix = dataUrl.slice(-100);
  return simpleHash(prefix + middle + suffix + dataUrl.length.toString());
}

/**
 * Simulates canvas LSB modification to produce a valid stego data URL
 */
function createMockStegoImage(coverDataUrl: string, payloadSize: number): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        resolve(coverDataUrl);
        return;
      }

      ctx.drawImage(img, 0, 0);
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imgData.data;

      // Seeded minor deterministic perturbations to simulate LSB embedding
      const step = Math.max(1, Math.floor((data.length / 4) / Math.max(10, payloadSize)));
      for (let i = 0; i < data.length && i < payloadSize * 32; i += step * 4) {
        // Toggle LSB slightly deterministically
        data[i] = (data[i] ^ 1); // Red LSB
        if (i + 1 < data.length) data[i + 1] = (data[i + 1] ^ ((i % 2) === 0 ? 1 : 0)); // Green LSB
      }

      ctx.putImageData(imgData, 0, 0);
      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = () => resolve(coverDataUrl);
    img.src = coverDataUrl;
  });
}

/**
 * Mock Handler: POST /embed
 */
export async function mockEmbedHandler(body: {
  coverDataUrl: string;
  payloadKind: 'text' | 'file' | 'image';
  rawPayload: string; // text string, file dataUrl, or secret image dataUrl
  payloadFilename?: string;
  payloadMimeType?: string;
  passphrase?: string;
  method?: string;
  coverMeta?: { width: number; height: number };
}): Promise<StegoResult> {
  const method = body.method || 'Adaptive-LSB-Spatial';
  const rawBytes = body.payloadKind === 'text'
    ? new TextEncoder().encode(body.rawPayload).length
    : Math.floor((body.rawPayload.length * 3) / 4); // base64 estimate

  const compressedBytes = Math.max(16, Math.floor(rawBytes * (body.payloadKind === 'text' ? 0.55 : 0.88)));
  const encryptedBytes = compressedBytes + 32; // AES-GCM IV + tag overhead

  const coverPixels = (body.coverMeta?.width || 800) * (body.coverMeta?.height || 600);
  const bpp = Number(((encryptedBytes * 8) / (coverPixels * 3)).toFixed(4));

  const stegoImageDataUrl = await createMockStegoImage(body.coverDataUrl, rawBytes);
  const fingerprint = getImageFingerprint(stegoImageDataUrl);
  const passphraseHash = simpleHash(body.passphrase || 'default-key');

  const payloadInfo: PayloadInfo = {
    kind: body.payloadKind,
    rawBytes,
    compressedBytes,
    encryptedBytes,
    isSimulated: true,
    filename: body.payloadFilename,
    mimeType: body.payloadMimeType,
    dataUrl: body.payloadKind === 'image' || body.payloadKind === 'file' ? body.rawPayload : undefined,
    textContent: body.payloadKind === 'text' ? body.rawPayload : undefined,
  };

  const record: SessionEmbedRecord = {
    id: 'emb_' + Date.now() + '_' + Math.floor(Math.random() * 1000),
    fingerprint,
    passphraseHash,
    payloadInfo,
    method,
    timestamp: Date.now(),
  };

  sessionRegistry.push(record);

  return {
    stegoImageDataUrl,
    embedTimeMs: Math.floor(250 + Math.min(600, rawBytes * 0.05)),
    method,
    bpp,
    isSimulated: true,
    payloadInfo,
    sessionToken: record.id,
  };
}

/**
 * Mock Handler: POST /extract
 */
export async function mockExtractHandler(body: {
  stegoImageDataUrl: string;
  passphrase?: string;
}): Promise<ExtractionResult> {
  if (!body.stegoImageDataUrl) {
    throw { code: 'INVALID_IMAGE', message: 'No steganographic image provided for extraction.' };
  }

  const fingerprint = getImageFingerprint(body.stegoImageDataUrl);
  const providedPassphraseHash = simpleHash(body.passphrase || 'default-key');

  // Look up in session registry
  // Match by exact fingerprint OR most recent session record if user re-uploaded the exact dataUrl
  let matchedRecord = sessionRegistry.find(r => r.fingerprint === fingerprint);

  if (!matchedRecord && sessionRegistry.length > 0) {
    // If exact fingerprint shifted slightly due to re-encoding, fallback to latest session record for seamless testing
    matchedRecord = sessionRegistry[sessionRegistry.length - 1];
  }

  if (!matchedRecord) {
    throw {
      code: 'NO_PAYLOAD_DETECTED',
      message: 'No steganographic payload header, carrier signature, or embedding markers detected in this image.'
    };
  }

  if (body.passphrase !== undefined && matchedRecord.passphraseHash !== providedPassphraseHash) {
    throw {
      code: 'AUTH_FAILED',
      message: 'Authentication failed. The provided passphrase does not match the encryption key or header signature.'
    };
  }

  const payload = matchedRecord.payloadInfo;

  return {
    kind: payload.kind,
    textContent: payload.textContent,
    fileDataUrl: payload.dataUrl,
    filename: payload.filename || (payload.kind === 'image' ? 'secret_extracted.png' : 'payload.dat'),
    mimeType: payload.mimeType || (payload.kind === 'image' ? 'image/png' : 'application/octet-stream'),
    fileSizeBytes: payload.rawBytes,
    extractTimeMs: Math.floor(180 + Math.random() * 80),
    integrityVerified: true,
    method: matchedRecord.method,
    isSimulated: true,
  };
}

/**
 * Mock Handler: POST /crypto/estimate
 */
export async function mockCryptoEstimateHandler(body: {
  rawBytes: number;
  algorithm?: string;
}): Promise<{ encryptedBytes: number; overheadBytes: number; isSimulated: boolean }> {
  const overheadBytes = 32; // 12-byte IV + 16-byte Auth Tag + 4-byte padding
  return {
    encryptedBytes: body.rawBytes + overheadBytes,
    overheadBytes,
    isSimulated: true,
  };
}

/**
 * Mock Handler: POST /compression/estimate
 */
export async function mockCompressionEstimateHandler(body: {
  rawBytes: number;
  kind: 'text' | 'file' | 'image';
}): Promise<{ compressedBytes: number; compressionRatio: number; isSimulated: boolean }> {
  let ratio = 0.5;
  if (body.kind === 'text') ratio = 0.45;
  else if (body.kind === 'file') ratio = 0.82;
  else if (body.kind === 'image') ratio = 0.95;

  const compressedBytes = Math.max(16, Math.floor(body.rawBytes * ratio));
  return {
    compressedBytes,
    compressionRatio: Number(ratio.toFixed(2)),
    isSimulated: true,
  };
}

/**
 * Mock Handler: POST /metrics/detect
 */
export async function mockDetectionScoreHandler(body: {
  bpp: number;
  textureScore?: number;
}): Promise<{ detectionScore: number; classifier: string; riskLevel: 'Low' | 'Moderate' | 'High'; isSimulated: boolean }> {
  const bpp = body.bpp || 0.1;
  const texture = body.textureScore || 0.5;

  // Higher bpp increases risk; higher texture decreases detectability
  const rawScore = (bpp * 1.5) * (1 - (texture * 0.45));
  const detectionScore = Math.max(0.02, Math.min(0.98, Number(rawScore.toFixed(3))));

  let riskLevel: 'Low' | 'Moderate' | 'High' = 'Low';
  if (detectionScore > 0.6) riskLevel = 'High';
  else if (detectionScore > 0.25) riskLevel = 'Moderate';

  return {
    detectionScore,
    classifier: 'Spatial-Rich-Model (SRM-39M) + Ensemble Classifier',
    riskLevel,
    isSimulated: true,
  };
}
