import React, { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertOctagon,
  // AlertTriangle,
  CheckCircle2,
  Cpu,
  Download,
  FileCode,
  FileText,
  Image as ImageIcon,
  KeyRound,
  Loader2,
  Lock,
} from 'lucide-react';

import { ImageUploader } from '../components/common/ImageUploader';
// import { CapacityMeter } from '../components/common/CapacityMeter';
import { RealMathBadge } from '../components/common/RealMathBadge';
import { useDemo } from '../context/DemoContext';
import { embedPayload } from '../services/steganography/stegoService';
// import { calculateCapacity } from '../utils/capacity';
import {
  extractImageData,
  formatBytes,
  readFileAsDataUrl,
} from '../utils/imageProcessing';
import { computeImageMetrics } from '../utils/metrics';
import { downloadDataUrl } from '../utils/download';
import { getSampleSecretImage } from '../utils/sampleData';
import { ImageQualityMetrics, StegoResult } from '../types';

type EmbedMode = 'text' | 'file' | 'image';

type UploadedPayload = {
  name: string;
  size: number;
  type: string;
  dataUrl: string;
};

export const EmbedPage: React.FC = () => {
  const navigate = useNavigate();

  const {
    coverDataUrl,
    coverMeta,
    secretImageDataUrl,
    payloadText,
    stegoResult,
    setCover,
    setPayloadText,
    setPayloadKind,
    setStegoResult,
  } = useDemo();

  const [activeMode, setActiveMode] = useState<EmbedMode>('text');
  const [textInput, setTextInput] = useState<string>(payloadText || '');
  const [uploadedFile, setUploadedFile] = useState<UploadedPayload | null>(null);
  const [secretImage, setSecretImage] = useState<UploadedPayload | null>(null);
  const [secretImageUrl, setSecretImageUrl] = useState<string>(
    secretImageDataUrl || getSampleSecretImage() || ''
  );

  const [passphrase, setPassphrase] = useState<string>('');
  const [confirmPassphrase, setConfirmPassphrase] = useState<string>('');

  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [embedError, setEmbedError] = useState<string | null>(null);
  const [localStegoResult, setLocalStegoResult] = useState<StegoResult | null>(
    stegoResult
  );
  const [metrics, setMetrics] = useState<ImageQualityMetrics | null>(null);

  const payloadBytes = useMemo(() => {
    if (activeMode === 'text') {
      return new TextEncoder().encode(textInput).length;
    }

    if (activeMode === 'file') {
      return uploadedFile?.size || 0;
    }

    if (activeMode === 'image') {
      if (secretImage) {
        return secretImage.size;
      }

      if (secretImageUrl.startsWith('data:')) {
        const base64Part = secretImageUrl.split(',')[1] || '';
        return Math.floor((base64Part.length * 3) / 4);
      }
    }

    return 0;
  }, [activeMode, textInput, uploadedFile, secretImage, secretImageUrl]);

  // const capacityResult = useMemo(() => {
  //   return calculateCapacity(coverMeta, payloadBytes);
  // }, [coverMeta, payloadBytes]);

  const changeMode = (mode: EmbedMode) => {
    setActiveMode(mode);
    setPayloadKind(mode);
    setEmbedError(null);
    setLocalStegoResult(null);
    setMetrics(null);
  };

  const handleTextChange = (value: string) => {
    setTextInput(value);
    setPayloadText(value);
  };

  const handleGeneralFile = async (file: File | undefined) => {
    if (!file) return;

    const dataUrl = await readFileAsDataUrl(file);

    setUploadedFile({
      name: file.name,
      size: file.size,
      type: file.type || 'application/octet-stream',
      dataUrl,
    });
  };

  const handleSecretImage = async (file: File | undefined) => {
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setEmbedError('The secret-image payload must be an image file.');
      return;
    }

    const dataUrl = await readFileAsDataUrl(file);

    const imagePayload: UploadedPayload = {
      name: file.name,
      size: file.size,
      type: file.type || 'image/png',
      dataUrl,
    };

    setSecretImage(imagePayload);
    setSecretImageUrl(dataUrl);
  };

  const validateInputs = (): string | null => {
    if (!coverDataUrl) {
      return 'Please provide a valid cover image.';
    }

    if (payloadBytes <= 0) {
      return 'Please enter text or select a payload file.';
    }

    if (!passphrase.trim()) {
      return 'Passphrase is required.';
    }

    if (passphrase.length < 8) {
      return 'Passphrase must contain at least 8 characters.';
    }

    if (!confirmPassphrase) {
      return 'Please confirm the passphrase.';
    }

    if (passphrase !== confirmPassphrase) {
      return 'Passphrases do not match.';
    }

    if (activeMode === 'file' && !uploadedFile) {
      return 'Please select a file payload.';
    }

    if (activeMode === 'image' && !secretImageUrl) {
      return 'Please select a secret image.';
    }

    return null;
  };

  const handleEmbed = async () => {
    const validationError = validateInputs();

    if (validationError) {
      setEmbedError(validationError);
      return;
    }

    // TypeScript safety: validateInputs already checks this.
    if (!coverDataUrl) return;

    setIsProcessing(true);
    setEmbedError(null);
    setLocalStegoResult(null);
    setMetrics(null);

    let rawPayload = '';
    let payloadFilename: string | undefined;
    let payloadMimeType: string | undefined;

    if (activeMode === 'text') {
      rawPayload = textInput;
    } else if (activeMode === 'file' && uploadedFile) {
      rawPayload = uploadedFile.dataUrl;
      payloadFilename = uploadedFile.name;
      payloadMimeType = uploadedFile.type;
    } else {
      rawPayload = secretImageUrl;
      payloadFilename = secretImage?.name || 'secret_image.png';
      payloadMimeType = secretImage?.type || 'image/png';
    }

    try {
      const result = await embedPayload({
        coverDataUrl,
        payloadKind: activeMode,
        rawPayload,
        payloadFilename,
        payloadMimeType,
        passphrase,
        method: 'GuardianPixel-HED-Adaptive-LSB',
        coverMeta: coverMeta
          ? {
              width: coverMeta.width,
              height: coverMeta.height,
            }
          : undefined,
      });

      if (result.isSimulated) {
        throw new Error(
          'The frontend received a simulated response. Start Flask and disable mock mode.'
        );
      }

      setLocalStegoResult(result);
      setStegoResult(result);

      const [coverImage, stegoImage] = await Promise.all([
        extractImageData(coverDataUrl),
        extractImageData(result.stegoImageDataUrl),
      ]);

      const calculatedMetrics = computeImageMetrics(
        coverImage.imageData,
        stegoImage.imageData,
        payloadBytes
      );

      setMetrics(calculatedMetrics);
    } catch (error: unknown) {
      const message =
        error instanceof Error
          ? error.message
          : 'GuardianPixel embedding failed.';

      setEmbedError(message);
    } finally {
      setIsProcessing(false);

      // Passphrases remain only for the current operation.
      setPassphrase('');
      setConfirmPassphrase('');
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Cpu className="w-6 h-6 text-brand-400" />
            <h1 className="text-2xl font-bold tracking-tight text-white">
              GuardianPixel Embedding
            </h1>
            <span className="inline-flex items-center rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-300">
              Real Flask Backend
            </span>
          </div>

          <p className="text-sm text-slate-400 max-w-3xl">
            Encrypt and embed text, files or an exact secret image using the
            real HED-guided adaptive LSB pipeline.
          </p>
        </div>

        {localStegoResult && (
          <button
            type="button"
            onClick={() => navigate('/extract')}
            className="px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white text-sm font-semibold border border-slate-700"
          >
            Continue to Extraction
          </button>
        )}
      </div>

      {embedError && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-start gap-3">
          <AlertOctagon className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Embedding halted</p>
            <p className="text-xs text-rose-300 mt-1">{embedError}</p>
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 pb-3">
        <button
          type="button"
          onClick={() => changeMode('text')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-xl border ${
            activeMode === 'text'
              ? 'bg-brand-600 text-white border-brand-500'
              : 'bg-slate-900 text-slate-400 border-slate-800'
          }`}
        >
          <FileText className="w-4 h-4" />
          Text
        </button>

        <button
          type="button"
          onClick={() => changeMode('file')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-xl border ${
            activeMode === 'file'
              ? 'bg-brand-600 text-white border-brand-500'
              : 'bg-slate-900 text-slate-400 border-slate-800'
          }`}
        >
          <FileCode className="w-4 h-4" />
          File
        </button>

        <button
          type="button"
          onClick={() => changeMode('image')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-xl border ${
            activeMode === 'image'
              ? 'bg-brand-600 text-white border-brand-500'
              : 'bg-slate-900 text-slate-400 border-slate-800'
          }`}
        >
          <ImageIcon className="w-4 h-4" />
          Exact Secret Image
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-7 flex flex-col gap-6">
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800">
            <h2 className="text-base font-semibold text-white mb-3">
              1. Carrier Cover Image
            </h2>

            <ImageUploader
              currentDataUrl={coverDataUrl}
              currentMeta={coverMeta}
              onImageSelected={(dataUrl, meta, title) =>
                setCover(dataUrl, meta, title)
              }
              onClear={() => setCover('', null as any, '')}
            />
          </div>

          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-4">
            <h2 className="text-base font-semibold text-white">
              2. Secret Payload
            </h2>

            {activeMode === 'text' && (
              <div>
                <label className="text-xs font-medium text-slate-300 block mb-1.5">
                  Confidential Message
                </label>

                <textarea
                  value={textInput}
                  onChange={(event) => handleTextChange(event.target.value)}
                  placeholder="Enter the message to hide..."
                  rows={6}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 font-mono focus:border-brand-500"
                />

                <div className="flex items-center justify-between text-xs text-slate-400 mt-1.5">
                  <span>
                    UTF-8 size:{' '}
                    <strong className="text-emerald-300 font-mono">
                      {formatBytes(payloadBytes)}
                    </strong>
                  </span>
                  <RealMathBadge label="Real UTF-8" />
                </div>
              </div>
            )}

            {activeMode === 'file' && (
              <div>
                <label className="text-xs font-medium text-slate-300 block mb-2">
                  File Payload
                </label>

                <input
                  type="file"
                  onChange={(event) =>
                    handleGeneralFile(event.target.files?.[0])
                  }
                  className="block w-full text-sm text-slate-300 file:mr-4 file:rounded-lg file:border-0 file:bg-indigo-600 file:px-4 file:py-2 file:text-white"
                />

                {uploadedFile && (
                  <p className="text-xs text-slate-400 mt-2">
                    {uploadedFile.name} — {formatBytes(uploadedFile.size)}
                  </p>
                )}
              </div>
            )}

            {activeMode === 'image' && (
              <div className="flex flex-col gap-3">
                <label className="text-xs font-medium text-slate-300">
                  Secret Image Payload
                </label>

                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/bmp"
                  onChange={(event) =>
                    handleSecretImage(event.target.files?.[0])
                  }
                  className="block w-full text-sm text-slate-300 file:mr-4 file:rounded-lg file:border-0 file:bg-brand-600 file:px-4 file:py-2 file:text-white"
                />

                {secretImageUrl && (
                  <div className="flex items-center gap-4 p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <img
                      src={secretImageUrl}
                      alt="Secret payload preview"
                      className="w-24 h-24 object-contain rounded-lg border border-slate-800"
                    />
                    <div className="text-xs text-slate-400">
                      <p className="text-white font-semibold">
                        {secretImage?.name || 'Sample secret image'}
                      </p>
                      <p>{formatBytes(payloadBytes)}</p>
                      <p className="mt-1">
                        Recovered as exact encrypted file bytes when capacity permits.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}

            <div className="pt-4 border-t border-slate-800">
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                  <KeyRound className="w-3.5 h-3.5 text-amber-400" />
                  Passphrase
                </label>

                <span className="text-[10px] text-emerald-400 font-mono flex items-center gap-1">
                  <Lock className="w-2.5 h-2.5" />
                  Never stored
                </span>
              </div>

              <input
                type="password"
                autoComplete="new-password"
                value={passphrase}
                onChange={(event) => setPassphrase(event.target.value)}
                placeholder="Minimum 8 characters..."
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 font-mono focus:border-brand-500"
              />

              <label className="text-xs font-medium text-slate-300 block mt-3 mb-1.5">
                Confirm Passphrase
              </label>

              <input
                type="password"
                autoComplete="new-password"
                value={confirmPassphrase}
                onChange={(event) => setConfirmPassphrase(event.target.value)}
                placeholder="Enter the same passphrase again..."
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 font-mono focus:border-brand-500"
              />

              {confirmPassphrase && passphrase !== confirmPassphrase && (
                <p className="text-xs text-rose-400 mt-1.5">
                  Passphrases do not match.
                </p>
              )}

              {confirmPassphrase && passphrase === confirmPassphrase && (
                <p className="text-xs text-emerald-400 mt-1.5">
                  Passphrases match.
                </p>
              )}
            </div>

            {/* {capacityResult.utilizationPct > 100 && (
              <div className="p-3.5 rounded-xl bg-rose-950/30 border border-rose-500/40 text-rose-300 text-xs flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold block">
                    Client estimate exceeds recommended capacity
                  </span>
                  <span>
                    The real Flask backend will perform the final capacity check.
                  </span>
                </div>
              </div>
            )} */}

            <button
              type="button"
              disabled={isProcessing || !coverDataUrl || payloadBytes === 0}
              onClick={handleEmbed}
              className={`w-full py-3 px-4 rounded-xl font-bold text-sm flex items-center justify-center gap-2 shadow-xl transition-all ${
                isProcessing || !coverDataUrl || payloadBytes === 0
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                  : 'bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-500 hover:to-indigo-500 text-white shadow-brand-500/20 active:scale-[0.98]'
              }`}
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running real HED and embedding pipeline...
                </>
              ) : (
                <>
                  <Cpu className="w-4 h-4" />
                  Generate Real Stego PNG
                </>
              )}
            </button>
          </div>
        </div>

        <div className="lg:col-span-5 flex flex-col gap-6">
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800">
        <p className="text-xs font-semibold text-slate-300">
          Real Capacity Source
        </p>

        <p className="text-xs text-slate-400 mt-1">
          Use the Compatibility page for the real HED-based cover
          category, encrypted requirement, one-channel capacity,
          selected-block percentage and recommendation.
        </p>
        </div>

          {localStegoResult ? (
            <div className="p-6 rounded-2xl bg-slate-900/90 border border-emerald-500/30 shadow-2xl flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <h3 className="text-base font-bold text-white">
                    Real Stego Image Generated
                  </h3>
                </div>

                <span className="text-[10px] px-2 py-1 rounded border border-emerald-500/30 bg-emerald-500/10 text-emerald-300">
                  isSimulated: false
                </span>
              </div>

              <div className="w-full h-56 rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center p-2">
                <img
                  src={localStegoResult.stegoImageDataUrl}
                  alt="GuardianPixel stego result"
                  className="max-w-full max-h-full object-contain"
                />
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">
                    Total BPP
                  </span>
                  <span className="font-mono font-bold text-white">
                    {Number(localStegoResult.bpp).toFixed(6)}
                  </span>
                </div>

                <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">
                    Backend Time
                  </span>
                  <span className="font-mono font-bold text-white">
                    {(localStegoResult.embedTimeMs / 1000).toFixed(2)} s
                  </span>
                </div>

                {metrics && (
                  <>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-400 block text-[10px]">
                        PSNR
                      </span>
                      <span className="font-mono font-bold text-emerald-300">
                        {Number(metrics.psnr).toFixed(2)} dB
                      </span>
                    </div>

                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-400 block text-[10px]">
                        SSIM
                      </span>
                      <span className="font-mono font-bold text-brand-300">
                        {Number(metrics.ssim).toFixed(6)}
                      </span>
                    </div>
                  </>
                )}
              </div>

              <button
                type="button"
                onClick={() =>
                  downloadDataUrl(
                    localStegoResult.stegoImageDataUrl,
                    `guardianpixel_stego_${Date.now()}.png`
                  )
                }
                className="w-full py-2.5 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs flex items-center justify-center gap-2"
              >
                <Download className="w-4 h-4" />
                Download Lossless Stego PNG
              </button>
            </div>
          ) : (
            <div className="p-8 rounded-2xl bg-slate-900/40 border border-dashed border-slate-800 text-center min-h-[260px] flex flex-col items-center justify-center">
              <Cpu className="w-8 h-8 text-slate-600 mb-2" />
              <p className="text-sm font-semibold text-slate-400">
                No stego image generated yet
              </p>
              <p className="text-xs text-slate-500 mt-1 max-w-xs">
                Select a cover, add a payload and enter matching passphrases.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
