import { SampleImageItem, PayloadInfo } from '../types';

/**
 * Creates high-quality procedural canvas test patterns with varying spatial frequencies and textures
 */
function createProceduralImage(
  width: number,
  height: number,
  type: 'natural' | 'urban' | 'texture' | 'smooth' | 'secret' | 'synthetic'
): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  if (type === 'texture') {
    // High-frequency noise + brick/weave texture
    const imgData = ctx.createImageData(width, height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const noise = (Math.sin(x * 0.2) * Math.cos(y * 0.2) + Math.sin(x * y * 0.005)) * 40;
        const grain = (Math.random() - 0.5) * 35;
        const base = 120 + noise + grain;
        imgData.data[idx] = Math.min(255, Math.max(0, base + 20)); // R
        imgData.data[idx + 1] = Math.min(255, Math.max(0, base)); // G
        imgData.data[idx + 2] = Math.min(255, Math.max(0, base - 25)); // B
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  } else if (type === 'natural') {
    // Mountainous landscape gradient with foliage noise
    const grad = ctx.createLinearGradient(0, 0, 0, height);
    grad.addColorStop(0, '#1e3a8a');
    grad.addColorStop(0.4, '#38bdf8');
    grad.addColorStop(0.7, '#15803d');
    grad.addColorStop(1, '#064e3b');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    // Forest canopy & mountain texture
    ctx.fillStyle = '#065f46';
    for (let i = 0; i < 60; i++) {
      ctx.beginPath();
      const cx = (i * 27) % width;
      const cy = height * 0.6 + Math.sin(i) * (height * 0.2);
      ctx.arc(cx, cy, 35 + (i % 15), 0, Math.PI * 2);
      ctx.fill();
    }
  } else if (type === 'urban') {
    // Architecture grid & building edges
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    // Skyscraper blocks with windows
    const buildingWidths = [45, 60, 50, 70, 55, 65, 80, 50];
    let curX = 10;
    for (let i = 0; i < buildingWidths.length && curX < width; i++) {
      const bWidth = buildingWidths[i];
      const bHeight = height * (0.45 + ((i * 17) % 45) / 100);
      const bY = height - bHeight;

      ctx.fillStyle = i % 2 === 0 ? '#1e293b' : '#334155';
      ctx.fillRect(curX, bY, bWidth, bHeight);

      // Window grid
      ctx.fillStyle = '#fde047';
      for (let wy = bY + 12; wy < height - 10; wy += 14) {
        for (let wx = curX + 6; wx < curX + bWidth - 6; wx += 10) {
          if ((wx + wy) % 3 !== 0) {
            ctx.fillRect(wx, wy, 5, 6);
          }
        }
      }
      curX += bWidth + 8;
    }
  } else if (type === 'smooth') {
    // Low-texture smooth studio gradient
    const grad = ctx.createRadialGradient(width * 0.5, height * 0.4, 20, width * 0.5, height * 0.5, width * 0.7);
    grad.addColorStop(0, '#f8fafc');
    grad.addColorStop(0.5, '#cbd5e1');
    grad.addColorStop(1, '#64748b');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);
  } else if (type === 'synthetic') {
    // Geometric test card
    ctx.fillStyle = '#18181b';
    ctx.fillRect(0, 0, width, height);
    for (let r = 20; r < Math.min(width, height) * 0.45; r += 20) {
      ctx.strokeStyle = r % 40 === 0 ? '#38bdf8' : '#f43f5e';
      ctx.lineWidth = 3;
      ctx.strokeRect(width / 2 - r, height / 2 - r, r * 2, r * 2);
    }
  } else if (type === 'secret') {
    // Secret QR / Security badge matrix
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(8, 8, width - 16, height - 16);
    ctx.fillStyle = '#000000';

    // Corner alignment squares
    const cornerSize = Math.floor(width * 0.28);
    ctx.fillRect(14, 14, cornerSize, cornerSize);
    ctx.clearRect(20, 20, cornerSize - 12, cornerSize - 12);
    ctx.fillRect(24, 24, cornerSize - 20, cornerSize - 20);

    ctx.fillRect(width - 14 - cornerSize, 14, cornerSize, cornerSize);
    ctx.clearRect(width - 8 - cornerSize, 20, cornerSize - 12, cornerSize - 12);
    ctx.fillRect(width - 4 - cornerSize, 24, cornerSize - 20, cornerSize - 20);

    // Center badge
    ctx.font = `bold ${Math.floor(width * 0.12)}px monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('CONFIDENTIAL', width / 2, height / 2);
    ctx.font = `600 ${Math.floor(width * 0.08)}px monospace`;
    ctx.fillText('KEY-SEC-8940', width / 2, height / 2 + 20);
  }

  return canvas.toDataURL('image/png');
}

// Pre-generated sample items (cached lazily)
let sampleCoverImagesCache: SampleImageItem[] | null = null;
let sampleSecretImageCache: string | null = null;

export function getSampleCoverImages(): SampleImageItem[] {
  if (sampleCoverImagesCache) return sampleCoverImagesCache;

  sampleCoverImagesCache = [
    {
      id: 'sample-texture',
      title: 'Dense Granite Texture',
      category: 'Texture-rich',
      description: 'High spatial frequency crystalline pattern. Exceptional steganographic capacity & lowest detectability.',
      expectedTexture: 'High',
      dataUrl: createProceduralImage(640, 480, 'texture'),
      previewColor: '#78716c',
      dimensions: { width: 640, height: 480 },
    },
    {
      id: 'sample-urban',
      title: 'Metropolitan Nightscape',
      category: 'Urban',
      description: 'High-contrast architectural edges, window grids, and rich structural diversity.',
      expectedTexture: 'High',
      dataUrl: createProceduralImage(640, 480, 'urban'),
      previewColor: '#1e293b',
      dimensions: { width: 640, height: 480 },
    },
    {
      id: 'sample-natural',
      title: 'Pine Forest & Horizon',
      category: 'Natural',
      description: 'Natural foliage variance with sky gradient transitions.',
      expectedTexture: 'Medium',
      dataUrl: createProceduralImage(640, 480, 'natural'),
      previewColor: '#047857',
      dimensions: { width: 640, height: 480 },
    },
    {
      id: 'sample-synthetic',
      title: 'Geometric Calibration Chart',
      category: 'Synthetic',
      description: 'Sharp vector lines and concentric boundaries for edge-detection evaluation.',
      expectedTexture: 'Medium',
      dataUrl: createProceduralImage(640, 480, 'synthetic'),
      previewColor: '#0369a1',
      dimensions: { width: 640, height: 480 },
    },
    {
      id: 'sample-smooth',
      title: 'Studio Portrait Gradient',
      category: 'Low-texture',
      description: 'Ultra-smooth vignette surface. Extremely vulnerable to LSB visual inspection & steganalysis.',
      expectedTexture: 'Low',
      dataUrl: createProceduralImage(640, 480, 'smooth'),
      previewColor: '#64748b',
      dimensions: { width: 640, height: 480 },
    },
  ];

  return sampleCoverImagesCache;
}

export function getSampleSecretImage(): string {
  if (sampleSecretImageCache) return sampleSecretImageCache;
  sampleSecretImageCache = createProceduralImage(180, 180, 'secret');
  return sampleSecretImageCache;
}

export const SAMPLE_TEXT_PAYLOADS = {
  short: 'Operation Aegis: Extraction waypoint alpha coordinates 37.7749 N, 122.4194 W. Passphrase rotation window 0600Z.',
  medium: `CLASSIFIED LAB DIRECTIVE // PROJECT GUARDIANPIXEL
==================================================
Date: 2026-08-14 | Clearance: Tier-3 Scientific

1. OBJECTIVE:
Deploy adaptive spatial LSB embedding within high-variance carrier regions to minimize SRM detection probability below 0.05.

2. PARAMETERS:
- Target Bit-Rate: 0.150 bpp
- Color Space: Rec. 709 sRGB
- Cipher: AES-256-GCM + 96-bit random IV + HMAC authentication
- Compression: Zlib RFC-1950 level-9 pre-pass

3. INTEGRITY ASSURANCE:
Pixel modifications constrained to local variance terciles ≥ 0.65 to ensure visual PSNR > 52.0 dB.`,
  code: `// Steganographic Channel Verification Script
function verifyStegoCarrier(data: Uint8Array, seed: number): boolean {
  let checksum = 0x811c9dc5;
  for (let i = 0; i < data.length; i++) {
    checksum ^= data[i];
    checksum = Math.imul(checksum, 0x01000193);
  }
  return (checksum >>> 0) === 0x4F9A2B11;
}`,
};

export interface UnifiedDemoState {
  coverImage: SampleImageItem;
  secretImage: string;
  sampleText: string;
  samplePassphrase: string;
  payloadInfo: PayloadInfo;
}

/**
 * Unified "Load Demo" action populating cover, secret image, text payload, and parameters
 */
export function getUnifiedDemoData(): UnifiedDemoState {
  const covers = getSampleCoverImages();
  const coverImage = covers[0]; // High texture rich granite
  const secretImage = getSampleSecretImage();
  const sampleText = SAMPLE_TEXT_PAYLOADS.medium;
  const samplePassphrase = 'GuardianPixel-Alpha-Key2026';

  const rawBytes = new TextEncoder().encode(sampleText).length;

  return {
    coverImage,
    secretImage,
    sampleText,
    samplePassphrase,
    payloadInfo: {
      kind: 'text',
      rawBytes,
      compressedBytes: Math.floor(rawBytes * 0.52),
      encryptedBytes: Math.floor(rawBytes * 0.52) + 32,
      isSimulated: true,
      textContent: sampleText,
    },
  };
}
