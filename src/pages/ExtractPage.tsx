import React, { useEffect, useState } from 'react';
import {
  AlertCircle,
  Check,
  Copy,
  Download,
  FileCode,
  FileKey,
  Image as ImageIcon,
  KeyRound,
  Loader2,
  Lock,
  ShieldCheck,
  Unlock,
} from 'lucide-react';

import { ImageUploader } from '../components/common/ImageUploader';
import { useDemo } from '../context/DemoContext';
import { extractPayload } from '../services/steganography/stegoService';
import { CoverImageMeta, ExtractionResult } from '../types';
import { downloadDataUrl, downloadText } from '../utils/download';
import { formatBytes } from '../utils/imageProcessing';

type RealExtractionResult = ExtractionResult & {
  sha256?: string;
};

export const ExtractPage: React.FC = () => {
  const { stegoResult } = useDemo();

  const [stegoDataUrl, setStegoDataUrl] = useState<string | null>(
    stegoResult?.stegoImageDataUrl || null
  );
  const [stegoMeta, setStegoMeta] = useState<CoverImageMeta | null>(null);
  const [passphrase, setPassphrase] = useState<string>('');
  const [isExtracting, setIsExtracting] = useState<boolean>(false);
  const [result, setResult] = useState<RealExtractionResult | null>(null);
  const [error, setError] = useState<{ code: string; message: string } | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  useEffect(() => {
    if (stegoResult?.stegoImageDataUrl) {
      setStegoDataUrl(stegoResult.stegoImageDataUrl);
      setResult(null);
      setError(null);
    }
  }, [stegoResult]);

  const handleExtract = async () => {
    if (!stegoDataUrl) {
      setError({
        code: 'MISSING_STEGO_IMAGE',
        message: 'Please upload the lossless GuardianPixel stego PNG.',
      });
      return;
    }

    if (!passphrase.trim()) {
      setError({
        code: 'MISSING_PASSPHRASE',
        message: 'Enter the passphrase used during embedding.',
      });
      return;
    }

    if (passphrase.length < 8) {
      setError({
        code: 'INVALID_PASSPHRASE',
        message: 'Passphrase must contain at least 8 characters.',
      });
      return;
    }

    setIsExtracting(true);
    setError(null);
    setResult(null);

    try {
      const response = (await extractPayload({
        stegoImageDataUrl: stegoDataUrl,
        passphrase,
      })) as RealExtractionResult;

      if (response.isSimulated) {
        throw new Error(
          'A simulated extraction response was received. Start Flask and disable mock mode.'
        );
      }

      setResult(response);
    } catch (caught: unknown) {
      const apiError = caught as Error & { code?: string };

      setError({
        code: apiError.code || 'AUTH_OR_DATA_FAILED',
        message:
          apiError.message ||
          'Extraction failed: wrong passphrase or modified image.',
      });
    } finally {
      setIsExtracting(false);
      setPassphrase('');
    }
  };

  const copyRecoveredText = async (text: string) => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-8">
      <div className="border-b border-slate-800 pb-6">
        <div className="flex items-center gap-2 mb-1">
          <FileKey className="w-6 h-6 text-brand-400" />
          <h1 className="text-2xl font-bold tracking-tight text-white">
            GuardianPixel Extraction
          </h1>
          <span className="inline-flex items-center rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-300">
            Real Flask Backend
          </span>
        </div>

        <p className="text-sm text-slate-400 max-w-4xl">
          Upload an unchanged lossless stego PNG and enter the passphrase once.
          Flask recovers the bootstrap, location map and encrypted payload, then
          verifies AES-GCM authentication and the SHA-256 payload digest.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-6 flex flex-col gap-6">
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-5">
            <h2 className="text-base font-semibold text-white">
              1. Lossless Stego PNG
            </h2>

            <ImageUploader
              label="Upload GuardianPixel Stego PNG"
              description="Only an unchanged PNG produced by GuardianPixel is supported."
              currentDataUrl={stegoDataUrl}
              currentMeta={stegoMeta}
              showSamplePresets={false}
              onImageSelected={(dataUrl, meta) => {
                if (meta.fileType.toUpperCase() !== 'PNG') {
                  setError({
                    code: 'LOSSY_STEGO_NOT_ALLOWED',
                    message:
                      'Extraction requires the original lossless stego PNG. JPEG, WebP and edited images are not supported.',
                  });
                  setStegoDataUrl(null);
                  setStegoMeta(null);
                  return;
                }

                setStegoDataUrl(dataUrl);
                setStegoMeta(meta);
                setError(null);
                setResult(null);
              }}
              onClear={() => {
                setStegoDataUrl(null);
                setStegoMeta(null);
                setResult(null);
                setError(null);
              }}
            />

            <div className="pt-4 border-t border-slate-800">
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                  <KeyRound className="w-3.5 h-3.5 text-amber-400" />
                  Decryption Passphrase
                </label>

                <span className="text-[10px] text-emerald-400 font-mono flex items-center gap-1">
                  <Lock className="w-2.5 h-2.5" />
                  Never stored or logged
                </span>
              </div>

              <input
                type="password"
                autoComplete="current-password"
                value={passphrase}
                onChange={(event) => setPassphrase(event.target.value)}
                placeholder="Enter the embedding passphrase once..."
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 font-mono focus:border-brand-500"
              />
            </div>

            <button
              type="button"
              disabled={isExtracting || !stegoDataUrl}
              onClick={handleExtract}
              className={`w-full py-3 px-4 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all ${
                isExtracting || !stegoDataUrl
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                  : 'bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-500 hover:to-indigo-500 text-white'
              }`}
            >
              {isExtracting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Reading map, authenticating and decrypting...
                </>
              ) : (
                <>
                  <Unlock className="w-4 h-4" />
                  Extract and Decrypt Real Payload
                </>
              )}
            </button>
          </div>
        </div>

        <div className="lg:col-span-6 flex flex-col gap-6">
          {error && (
            <div className="p-6 rounded-2xl bg-rose-500/10 border border-rose-500/40 text-rose-200 flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-rose-400" />
                <h3 className="font-bold">Extraction Failed</h3>
                <span className="ml-auto text-[10px] font-mono px-2 py-1 rounded bg-rose-950 border border-rose-800">
                  {error.code}
                </span>
              </div>

              <p className="text-sm">{error.message}</p>

              <p className="text-xs text-slate-400 bg-slate-950/70 border border-slate-800 rounded-lg p-3">
                GuardianPixel intentionally does not reveal whether the password,
                metadata or encrypted payload failed. No partial plaintext is returned.
              </p>
            </div>
          )}

          {result && (
            <div className="p-6 rounded-2xl bg-slate-900/90 border border-emerald-500/30 flex flex-col gap-5">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-5 h-5 text-emerald-400" />
                  <h3 className="font-bold text-white">
                    Payload Successfully Recovered
                  </h3>
                </div>

                <span className="text-[10px] px-2 py-1 rounded border border-emerald-500/30 bg-emerald-500/10 text-emerald-300">
                  Backend Verified
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Payload type</p>
                  <p className="font-mono font-bold text-white uppercase">
                    {result.kind}
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Recovered size</p>
                  <p className="font-mono font-bold text-white">
                    {formatBytes(result.fileSizeBytes || 0)}
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Integrity</p>
                  <p className="font-mono font-bold text-emerald-300">
                    {result.integrityVerified
                      ? 'AES-GCM + SHA-256 OK'
                      : 'Unverified'}
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Backend time</p>
                  <p className="font-mono font-bold text-white">
                    {(result.extractTimeMs / 1000).toFixed(2)} s
                  </p>
                </div>
              </div>

              {result.sha256 && (
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-[10px] uppercase text-slate-500 mb-1">
                    Recovered payload SHA-256
                  </p>
                  <p className="text-[11px] text-slate-300 font-mono break-all">
                    {result.sha256}
                  </p>
                </div>
              )}

              {result.kind === 'text' && result.textContent && (
                <div className="flex flex-col gap-2">
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-slate-400">Recovered plaintext</p>
                    <button
                      type="button"
                      onClick={() => copyRecoveredText(result.textContent!)}
                      className="text-xs text-brand-400 flex items-center gap-1"
                    >
                      {copied ? (
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                      {copied ? 'Copied' : 'Copy'}
                    </button>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-sm text-slate-200 whitespace-pre-wrap max-h-64 overflow-y-auto">
                    {result.textContent}
                  </div>

                  <button
                    type="button"
                    onClick={() =>
                      downloadText(result.textContent!, 'guardianpixel_recovered.txt')
                    }
                    className="w-full py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold flex items-center justify-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download Recovered Text
                  </button>
                </div>
              )}

              {result.kind === 'image' && result.fileDataUrl && (
                <div className="flex flex-col gap-3">
                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <ImageIcon className="w-4 h-4" />
                    Exact recovered secret image
                  </div>

                  <div className="h-60 bg-slate-950 border border-slate-800 rounded-xl p-2 flex items-center justify-center">
                    <img
                      src={result.fileDataUrl}
                      alt="Recovered secret payload"
                      className="max-h-full max-w-full object-contain"
                    />
                  </div>

                  <button
                    type="button"
                    onClick={() =>
                      downloadDataUrl(
                        result.fileDataUrl!,
                        result.filename || 'guardianpixel_recovered_image.png'
                      )
                    }
                    className="w-full py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold flex items-center justify-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download Recovered Image
                  </button>
                </div>
              )}

              {result.kind === 'file' && result.fileDataUrl && (
                <div className="flex flex-col gap-3">
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center gap-3">
                    <FileCode className="w-6 h-6 text-indigo-400" />
                    <div>
                      <p className="font-semibold text-white">
                        {result.filename || 'Recovered file'}
                      </p>
                      <p className="text-xs text-slate-400">
                        {result.mimeType || 'application/octet-stream'}
                      </p>
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={() =>
                      downloadDataUrl(
                        result.fileDataUrl!,
                        result.filename || 'guardianpixel_recovered_file.bin'
                      )
                    }
                    className="w-full py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold flex items-center justify-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download Recovered File
                  </button>
                </div>
              )}
            </div>
          )}

          {!error && !result && (
            <div className="p-8 rounded-2xl bg-slate-900/40 border border-dashed border-slate-800 text-center min-h-[320px] flex flex-col items-center justify-center">
              <FileKey className="w-9 h-9 text-slate-600 mb-3" />
              <p className="text-sm font-semibold text-slate-400">
                Ready for real extraction
              </p>
              <p className="text-xs text-slate-500 max-w-sm mt-2">
                Upload the unchanged stego PNG and enter the correct passphrase.
                HED is not required on the receiver side.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
