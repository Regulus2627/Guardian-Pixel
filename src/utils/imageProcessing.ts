import { CoverImageMeta } from '../types';

/**
 * Loads an image from a URL or DataURL into an HTMLImageElement
 */
export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load image resource'));
    img.src = src;
  });
}

/**
 * Reads a File object as Data URL
 */
export function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error('Failed to read file contents'));
    reader.readAsDataURL(file);
  });
}

/**
 * Extracts real cover image metadata from a File and its Image element
 */
export async function extractCoverMetadata(file: File, dataUrl: string): Promise<CoverImageMeta> {
  const img = await loadImage(dataUrl);

  const fileType = file.type || (file.name.endsWith('.png') ? 'image/png' : 'image/jpeg');
  const fileSizeBytes = file.size;

  return {
    width: img.width,
    height: img.height,
    colorMode: 'RGB (sRGB)',
    fileType: fileType.toUpperCase().replace('IMAGE/', ''),
    fileSizeBytes,
    channels: 3, // Standard RGB
    bitDepth: 8, // 8-bit standard per channel
  };
}

/**
 * Extracts metadata directly from a Data URL (e.g. for sample images)
 */
export async function extractDataUrlMetadata(dataUrl: string, _title = 'Sample Image'): Promise<CoverImageMeta> {
  const img = await loadImage(dataUrl);

  // Approximate byte length from base64 string length
  const base64Length = dataUrl.split(',')[1]?.length || dataUrl.length;
  const fileSizeBytes = Math.floor((base64Length * 3) / 4);

  const isPng = dataUrl.startsWith('data:image/png');
  const fileType = isPng ? 'PNG' : 'JPEG';

  return {
    width: img.width,
    height: img.height,
    colorMode: 'RGB (sRGB)',
    fileType,
    fileSizeBytes,
    channels: 3,
    bitDepth: 8,
  };
}

/**
 * Extracts raw ImageData from an image source, downscaling if dimensions exceed maxDimension
 */
export async function extractImageData(
  src: string,
  maxDimension = 2000
): Promise<{ imageData: ImageData; width: number; height: number; scale: number }> {
  const img = await loadImage(src);

  let width = img.width;
  let height = img.height;
  let scale = 1.0;

  if (Math.max(width, height) > maxDimension) {
    scale = maxDimension / Math.max(width, height);
    width = Math.round(width * scale);
    height = Math.round(height * scale);
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('Canvas 2D context not available');
  }

  ctx.drawImage(img, 0, 0, width, height);
  const imageData = ctx.getImageData(0, 0, width, height);

  return { imageData, width, height, scale };
}

/**
 * Formats byte size into human readable string
 */
export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}
