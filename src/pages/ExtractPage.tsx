import React, { useState, useEffect } from 'react';
import { useDemo } from '../context/DemoContext';
import { ImageUploader } from '../components/common/ImageUploader';
import { SimulatedBadge } from '../components/common/SimulatedBadge';
import { extractPayload } from '../services/steganography/stegoService';
import { ExtractionResult, CoverImageMeta } from '../types';
import { downloadDataUrl, downloadText } from '../utils/download';
import { formatBytes } from '../utils/imageProcessing';
import {
  FileKey,
  KeyRound,
  Lock,
  Unlock,
  Download,
  FileCode,
  Loader2,
  ShieldCheck,
  ShieldAlert,
  Copy,
  Check,
} from 'lucide-react';

export const ExtractPage: React.FC = () => {
  const { stegoResult, passphrase: initialPassphrase } = useDemo();

  const [stegoDataUrl, setStegoDataUrl] = useState<string | null>(stegoResult?.stegoImageDataUrl || null);
  const [stegoMeta, setStegoMeta] = useState<CoverImageMeta | null>(null);
  const [passphrase, setPassphraseInput] = useState<string>(initialPassphrase || '');
  const [isExtracting, setIsExtracting] = useState<boolean>(false);
  const [extractionResult, setExtractionResult] = useState<ExtractionResult | null>(null);
  const [extractError, setExtractError] = useState<{ code: string; message: string } | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  // Sync if stegoResult changes
  useEffect(() => {
    if (stegoResult?.stegoImageDataUrl) {
      setStegoDataUrl(stegoResult.stegoImageDataUrl);
    }
  }, [stegoResult]);

  const handleExtract = async () => {
    if (!stegoDataUrl) {
      setExtractError({
        code: 'MISSING_IMAGE',
        message: 'Please provide a steganographic carrier image to extract payload from.',
      });
      return;
    }

    if (!passphrase.trim()) {
      setExtractError({
        code: 'MISSING_KEY',
        message: 'Passphrase is required for decryption and authentication verification.',
      });
      return;
    }

    setIsExtracting(true);
    setExtractError(null);
    setExtractionResult(null);

    try {
      const result = await extractPayload({
        stegoImageDataUrl: stegoDataUrl,
        passphrase,
      });

      setExtractionResult(result);
    } catch (err: any) {
      const code = err.code || 'EXTRACTION_ERROR';
      let message = err.message || 'Extraction pipeline failed.';

      // Format clean, non-leaking user-friendly messages for each error class
      if (code === 'AUTH_FAILED') {
        message = 'Authentication failed: Passphrase incorrect or integrity tag mismatch.';
      } else if (code === 'NO_PAYLOAD_DETECTED') {
        message = 'No steganographic header, carrier signature, or embedding markers detected in this carrier.';
      } else if (code === 'CORRUPTED') {
        message = 'Payload integrity check failed: Data was modified or corrupted in transit.';
      }

      setExtractError({ code, message });
    } finally {
      setIsExtracting(false);
      // Security hygiene: zero passphrase from component memory
      setPassphraseInput('');
    }
  };

  const handleCopyText = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-850 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <FileKey className="w-6 h-6 text-brand-400" />
            <h1 className="text-2xl font-bold tracking-tight text-white">Steganographic Extraction Suite</h1>
            <SimulatedBadge label="Simulated Engine" />
          </div>
          <p className="text-sm text-slate-400 max-w-3xl">
            Authenticate, decrypt, and recover secret payloads from stego carriers. Validates tamper integrity with strict error isolation.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Stego Carrier & Key Input */}
        <div className="lg:col-span-6 flex flex-col gap-6">
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-4">
            <h2 className="text-base font-semibold text-white">1. Steganographic Carrier Image</h2>

            <ImageUploader
              label="Upload Stego Carrier Image"
              description="PNG, WEBP, BMP or JPG suspected of containing hidden data"
              currentDataUrl={stegoDataUrl}
              currentMeta={stegoMeta}
              onImageSelected={(dataUrl, meta) => {
                setStegoDataUrl(dataUrl);
                setStegoMeta(meta);
                setExtractError(null);
                setExtractionResult(null);
              }}
              onClear={() => {
                setStegoDataUrl(null);
                setStegoMeta(null);
                setExtractionResult(null);
              }}
            />

            {/* Passphrase Input */}
            <div className="pt-4 border-t border-slate-800">
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                  <KeyRound className="w-3.5 h-3.5 text-amber-400" />
                  <span>Decryption Passphrase</span>
                </label>
                <span className="text-[10px] text-emerald-400 font-mono flex items-center gap-1">
                  <Lock className="w-2.5 h-2.5" />
                  Passphrase Never Logged
                </span>
              </div>

              <input
                type="password"
                autoComplete="new-password"
                value={passphrase}
                onChange={(e) => setPassphraseInput(e.target.value)}
                placeholder="Enter payload encryption passphrase..."
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 font-mono focus:border-brand-500"
              />
            </div>

            {/* Extract Action Button */}
            <button
              type="button"
              disabled={isExtracting || !stegoDataUrl}
              onClick={handleExtract}
              className={`w-full py-3 px-4 rounded-xl font-bold text-sm flex items-center justify-center gap-2 shadow-xl transition-all ${
                isExtracting || !stegoDataUrl
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                  : 'bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-500 hover:to-indigo-500 text-white shadow-brand-500/20 active:scale-[0.98]'
              }`}
            >
              {isExtracting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin text-white" />
                  <span>Verifying Headers & Extracting Payload...</span>
                </>
              ) : (
                <>
                  <Unlock className="w-4 h-4 text-brand-300" />
                  <span>Extract & Decrypt Payload</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Extraction Result or Safe Error State */}
        <div className="lg:col-span-6 flex flex-col gap-6">
          {extractError && (
            <div className="p-6 rounded-2xl bg-rose-950/30 border border-rose-500/40 text-rose-200 shadow-xl flex flex-col gap-3">
              <div className="flex items-center gap-2.5">
                <ShieldAlert className="w-5 h-5 text-rose-400" />
                <h3 className="font-bold text-base text-rose-300">Extraction Denied / Failed</h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-900/60 border border-rose-700 text-rose-300 ml-auto">
                  {extractError.code}
                </span>
              </div>

              <p className="text-sm text-rose-300 leading-relaxed">
                {extractError.message}
              </p>

              <div className="p-3 rounded-xl bg-slate-950/80 border border-rose-950 text-xs text-slate-400">
                <span className="font-semibold text-slate-300 block mb-1">Security Isolation Protocol:</span>
                The payload extraction interface refuses to reveal cryptographic internals, bit offsets, or partial decrypted fragments when authentication or integrity checks fail.
              </div>
            </div>
          )}

          {extractionResult && (
            <div className="p-6 rounded-2xl bg-slate-900/90 border border-emerald-500/30 shadow-2xl flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-5 h-5 text-emerald-400" />
                  <h3 className="text-base font-bold text-white">Payload Successfully Recovered</h3>
                </div>
                <SimulatedBadge />
              </div>

              {/* Status Header */}
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 block text-[10px]">Payload Type</span>
                  <span className="font-semibold text-slate-200 uppercase font-mono">
                    {extractionResult.kind}
                  </span>
                </div>

                <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 block text-[10px]">Recovered Size</span>
                  <span className="font-mono font-semibold text-emerald-300 tabular-nums">
                    {formatBytes(extractionResult.fileSizeBytes || 0)}
                  </span>
                </div>

                <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 block text-[10px]">Integrity Check</span>
                  <span className="font-mono font-semibold text-teal-300">
                    {extractionResult.integrityVerified ? 'HMAC-SHA256 OK' : 'Unverified'}
                  </span>
                </div>
              </div>

              {/* Text Payload View */}
              {extractionResult.kind === 'text' && extractionResult.textContent && (
                <div className="flex flex-col gap-2">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Decrypted Plaintext:</span>
                    <button
                      type="button"
                      onClick={() => handleCopyText(extractionResult.textContent!)}
                      className="inline-flex items-center gap-1 text-xs text-brand-400 hover:text-brand-300 font-medium"
                    >
                      {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>{copied ? 'Copied' : 'Copy Text'}</span>
                    </button>
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-slate-200 max-h-56 overflow-y-auto whitespace-pre-wrap">
                    {extractionResult.textContent}
                  </div>

                  <button
                    type="button"
                    onClick={() => downloadText(extractionResult.textContent!, 'recovered_payload.txt')}
                    className="w-full mt-2 py-2.5 px-4 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-md shadow-brand-500/15"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download Plaintext File (.txt)</span>
                  </button>
                </div>
              )}

              {/* Image Payload View */}
              {extractionResult.kind === 'image' && extractionResult.fileDataUrl && (
                <div className="flex flex-col gap-2">
                  <span className="text-xs text-slate-400">Recovered Secret Image:</span>
                  <div className="w-full h-52 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-center p-2">
                    <img
                      src={extractionResult.fileDataUrl}
                      alt="Recovered Secret"
                      className="max-w-full max-h-full object-contain"
                    />
                  </div>

                  <button
                    type="button"
                    onClick={() => downloadDataUrl(extractionResult.fileDataUrl!, 'recovered_secret.png')}
                    className="w-full mt-2 py-2.5 px-4 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-md shadow-brand-500/15"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download Recovered Image (.png)</span>
                  </button>
                </div>
              )}

              {/* File Payload View */}
              {extractionResult.kind === 'file' && (
                <div className="flex flex-col gap-2">
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center gap-3">
                    <FileCode className="w-6 h-6 text-indigo-400" />
                    <div>
                      <span className="font-semibold text-sm text-white block">{extractionResult.filename}</span>
                      <span className="text-xs text-slate-400">{extractionResult.mimeType}</span>
                    </div>
                  </div>

                  {extractionResult.fileDataUrl && (
                    <button
                      type="button"
                      onClick={() => downloadDataUrl(extractionResult.fileDataUrl!, extractionResult.filename || 'extracted_payload.dat')}
                      className="w-full mt-2 py-2.5 px-4 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-md shadow-brand-500/15"
                    >
                      <Download className="w-4 h-4" />
                      <span>Download Extracted Binary Payload</span>
                    </button>
                  )}
                </div>
              )}
            </div>
          )}

          {!extractError && !extractionResult && (
            <div className="p-8 rounded-2xl bg-slate-900/40 border border-dashed border-slate-800 text-center flex flex-col items-center justify-center min-h-[280px]">
              <FileKey className="w-8 h-8 text-slate-600 mb-2" />
              <p className="text-sm font-semibold text-slate-400">Ready for Carrier Extraction</p>
              <p className="text-xs text-slate-500 max-w-xs mt-1">
                Upload a steganographic carrier image, enter the encryption key, and click Extract to inspect the recovered contents.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
