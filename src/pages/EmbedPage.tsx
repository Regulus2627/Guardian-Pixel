import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDemo } from '../context/DemoContext';
import { ImageUploader } from '../components/common/ImageUploader';
import { CapacityMeter } from '../components/common/CapacityMeter';
import { SimulatedBadge } from '../components/common/SimulatedBadge';
import { RealMathBadge } from '../components/common/RealMathBadge';
import { embedPayload } from '../services/steganography/stegoService';
import { calculateCapacity } from '../utils/capacity';
import { extractImageData, formatBytes, readFileAsDataUrl } from '../utils/imageProcessing';
import { computeImageMetrics, renderDifferenceCanvas } from '../utils/metrics';
import { downloadDataUrl } from '../utils/download';
import { getSampleSecretImage } from '../utils/sampleData';
import {
  Cpu,
  Lock,
  Download,
  CheckCircle2,
  AlertOctagon,
  AlertTriangle,
  ArrowRight,
  FileText,
  FileCode,
  Image as ImageIcon,
  Loader2,
  KeyRound,
} from 'lucide-react';
import { ImageQualityMetrics, StegoResult } from '../types';

type EmbedMode = 'text' | 'file' | 'image';

export const EmbedPage: React.FC = () => {
  const navigate = useNavigate();
  const {
    coverDataUrl,
    coverMeta,
    secretImageDataUrl,
    payloadText,
    passphrase: initialPassphrase,
    stegoResult,
    setCover,
    setStegoResult,
  } = useDemo();

  const [activeMode, setActiveMode] = useState<EmbedMode>('text');
  const [textInput, setTextInput] = useState<string>(payloadText || '');
  const [passphrase, setPassphraseInput] = useState<string>(initialPassphrase || '');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [embedError, setEmbedError] = useState<string | null>(null);

  // File mode state
  const [uploadedFile, setUploadedFile] = useState<{
    name: string;
    size: number;
    type: string;
    dataUrl: string;
  } | null>(null);

  // Secret image mode state
  const [secretImgUrl, setSecretImgUrl] = useState<string>(secretImageDataUrl || getSampleSecretImage());
  const [secretImgSize, setSecretImgSize] = useState<number>(0);

  // Embedding pipeline result & metrics
  const [localStegoResult, setLocalStegoResult] = useState<StegoResult | null>(stegoResult);
  const [metrics, setMetrics] = useState<ImageQualityMetrics | null>(null);

  // Secret diff canvas ref for exact-image mode
  const secretDiffCanvasRef = useRef<HTMLCanvasElement>(null);

  // Compute raw payload bytes
  const payloadBytes = useMemo(() => {
    if (activeMode === 'text') {
      return new TextEncoder().encode(textInput).length;
    } else if (activeMode === 'file') {
      return uploadedFile?.size || 0;
    } else if (activeMode === 'image') {
      return secretImgSize || Math.floor((secretImgUrl.length * 3) / 4);
    }
    return 0;
  }, [activeMode, textInput, uploadedFile, secretImgUrl, secretImgSize]);

  // Real capacity evaluation
  const capacityResult = useMemo(() => {
    return calculateCapacity(coverMeta, payloadBytes);
  }, [coverMeta, payloadBytes]);

  // Handle secret image loading & sizing
  useEffect(() => {
    if (secretImgUrl) {
      const img = new Image();
      img.onload = () => {
        const byteCount = img.width * img.height * 3;
        setSecretImgSize(byteCount);
      };
      img.src = secretImgUrl;
    }
  }, [secretImgUrl]);

  // Handle file input for General File mode
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const dataUrl = await readFileAsDataUrl(file);
      setUploadedFile({
        name: file.name,
        size: file.size,
        type: file.type || 'application/octet-stream',
        dataUrl,
      });
    }
  };

  // Run Embedding Pipeline
  const handleEmbed = async () => {
    if (!coverDataUrl) {
      setEmbedError('Please provide a valid cover carrier image first.');
      return;
    }

    if (payloadBytes === 0) {
      setEmbedError('Payload is empty. Please enter text, upload a file, or select a secret image.');
      return;
    }

    if (!passphrase.trim()) {
      setEmbedError('Passphrase is required to encrypt and authenticate the steganographic payload.');
      return;
    }

    setIsProcessing(true);
    setEmbedError(null);

    let rawPayloadContent = '';
    let payloadFilename: string | undefined;
    let payloadMimeType: string | undefined;

    if (activeMode === 'text') {
      rawPayloadContent = textInput;
    } else if (activeMode === 'file' && uploadedFile) {
      rawPayloadContent = uploadedFile.dataUrl;
      payloadFilename = uploadedFile.name;
      payloadMimeType = uploadedFile.type;
    } else if (activeMode === 'image') {
      rawPayloadContent = secretImgUrl;
      payloadFilename = 'secret_image.png';
      payloadMimeType = 'image/png';
    }

    try {
      const result = await embedPayload({
        coverDataUrl,
        payloadKind: activeMode,
        rawPayload: rawPayloadContent,
        payloadFilename,
        payloadMimeType,
        passphrase,
        method: activeMode === 'image' ? 'StegEx-ExactNet' : 'Adaptive-LSB-Spatial',
        coverMeta: coverMeta ? { width: coverMeta.width, height: coverMeta.height } : undefined,
      });

      setLocalStegoResult(result);
      setStegoResult(result);

      // Compute real PSNR, SSIM, MSE against original cover image
      const [coverImgData, stegoImgData] = await Promise.all([
        extractImageData(coverDataUrl),
        extractImageData(result.stegoImageDataUrl),
      ]);

      const realMetrics = computeImageMetrics(coverImgData.imageData, stegoImgData.imageData, payloadBytes);
      setMetrics(realMetrics);

      // If exact image mode, render real secret difference canvas
      if (activeMode === 'image' && secretDiffCanvasRef.current) {
        const [origSecret, extractedSecret] = await Promise.all([
          extractImageData(secretImgUrl),
          extractImageData(result.payloadInfo.dataUrl || secretImgUrl),
        ]);
        renderDifferenceCanvas(secretDiffCanvasRef.current, origSecret.imageData, extractedSecret.imageData, {
          amplification: 20,
        });
      }
    } catch (err: any) {
      setEmbedError(err.message || 'Steganographic embedding pipeline encountered an error.');
    } finally {
      setIsProcessing(false);
      // Passphrase hygiene: clear local state as required by security guidelines
      setPassphraseInput('');
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-850 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Cpu className="w-6 h-6 text-brand-400" />
            <h1 className="text-2xl font-bold tracking-tight text-white">Steganographic Embedding Suite</h1>
            <SimulatedBadge label="Simulated Engine" />
          </div>
          <p className="text-sm text-slate-400 max-w-3xl">
            Configure payload preparation, ephemeral encryption, capacity safety constraints, and generate carrier stego images with real-time fidelity metrics.
          </p>
        </div>

        {localStegoResult && (
          <button
            type="button"
            onClick={() => navigate('/compare')}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-semibold text-sm border border-slate-700 transition-all self-start md:self-auto"
          >
            <span>Inspect in Compare View</span>
            <ArrowRight className="w-4 h-4 text-brand-400" />
          </button>
        )}
      </div>

      {embedError && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-start gap-3">
          <AlertOctagon className="w-5 h-5 flex-shrink-0 text-rose-400 mt-0.5" />
          <div>
            <p className="font-semibold">Embedding Halted</p>
            <p className="text-xs text-rose-400 mt-0.5">{embedError}</p>
          </div>
        </div>
      )}

      {/* Mode Selector Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
        <button
          type="button"
          onClick={() => setActiveMode('text')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-xl transition-all ${
            activeMode === 'text'
              ? 'bg-brand-600 text-white shadow-md shadow-brand-500/20'
              : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
          }`}
        >
          <FileText className="w-4 h-4" />
          <span>Mode 1: Confidential Text</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveMode('file')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-xl transition-all ${
            activeMode === 'file'
              ? 'bg-brand-600 text-white shadow-md shadow-brand-500/20'
              : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
          }`}
        >
          <FileCode className="w-4 h-4" />
          <span>Mode 2: General File</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveMode('image')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-xl transition-all ${
            activeMode === 'image'
              ? 'bg-brand-600 text-white shadow-md shadow-brand-500/20'
              : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
          }`}
        >
          <ImageIcon className="w-4 h-4" />
          <span>Mode 3: Exact Secret Image</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Form: Cover & Payload inputs */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          {/* Carrier Image Selector */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800">
            <h2 className="text-base font-semibold text-white mb-3">1. Carrier Cover Image</h2>
            <ImageUploader
              currentDataUrl={coverDataUrl}
              currentMeta={coverMeta}
              onImageSelected={(dataUrl, meta, title) => setCover(dataUrl, meta, title)}
              onClear={() => setCover('', null as any, '')}
            />
          </div>

          {/* Mode-Specific Payload Inputs */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-4">
            <h2 className="text-base font-semibold text-white">2. Payload Configuration</h2>

            {activeMode === 'text' && (
              <div>
                <label className="text-xs font-medium text-slate-300 block mb-1.5">
                  Plaintext Confidential Message
                </label>
                <textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Enter message to embed into cover image carrier..."
                  rows={5}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 font-mono focus:border-brand-500"
                />
                <div className="flex items-center justify-between text-xs text-slate-400 mt-1">
                  <span>UTF-8 Size: <strong className="text-emerald-300 font-mono">{formatBytes(payloadBytes)}</strong></span>
                  <RealMathBadge label="Real UTF-8" />
                </div>
              </div>
            )}

            {activeMode === 'file' && (
              <div className="flex flex-col gap-3">
                <label className="text-xs font-medium text-slate-300">Upload Target File (Document, Binary, Key)</label>
                <div className="p-6 border-2 border-dashed border-slate-700 rounded-xl bg-slate-950 text-center">
                  <input
                    type="file"
                    onChange={handleFileUpload}
                    id="general-file-input"
                    className="hidden"
                  />
                  <label htmlFor="general-file-input" className="cursor-pointer">
                    <FileCode className="w-8 h-8 text-indigo-400 mx-auto mb-2" />
                    <span className="text-xs font-medium text-slate-200 block">
                      {uploadedFile ? uploadedFile.name : 'Click to select arbitrary file'}
                    </span>
                    <span className="text-[11px] text-slate-500 mt-0.5 block">
                      {uploadedFile ? `${formatBytes(uploadedFile.size)} (${uploadedFile.type})` : 'Any binary format up to safe capacity'}
                    </span>
                  </label>
                </div>
              </div>
            )}

            {activeMode === 'image' && (
              <div className="flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium text-slate-300">Secret Image to Hide</label>
                  <span className="text-xs font-mono text-slate-400">{formatBytes(payloadBytes)} payload</span>
                </div>

                <div className="flex items-center gap-4 p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="w-24 h-24 rounded-lg bg-slate-900 border border-slate-800 overflow-hidden flex items-center justify-center">
                    <img src={secretImgUrl} alt="Secret preview" className="w-full h-full object-contain" />
                  </div>
                  <div className="text-xs text-slate-300 flex-1">
                    <span className="font-semibold text-white block mb-1">Confidential Key Matrix</span>
                    <p className="text-slate-400 text-[11px] mb-2">
                      Exact pixel reconstruction mode embeds high-entropy secret imagery losslessly into the cover.
                    </p>
                    <button
                      type="button"
                      onClick={() => setSecretImgUrl(getSampleSecretImage())}
                      className="text-xs text-brand-400 hover:underline font-medium"
                    >
                      Reload Default Sample Secret
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Passphrase Input with strict security rules */}
            <div className="pt-3 border-t border-slate-800">
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                  <KeyRound className="w-3.5 h-3.5 text-amber-400" />
                  <span>Encryption & Authentication Passphrase</span>
                </label>
                <span className="text-[10px] text-emerald-400 font-mono flex items-center gap-1">
                  <Lock className="w-2.5 h-2.5" />
                  Ephemeral (Never Saved)
                </span>
              </div>

              <input
                type="password"
                autoComplete="new-password"
                value={passphrase}
                onChange={(e) => setPassphraseInput(e.target.value)}
                placeholder="Enter strong passphrase for key derivation..."
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 font-mono focus:border-brand-500"
              />
            </div>

            {/* Capacity Safety Warning if overloaded */}
            {capacityResult.utilizationPct > 100 && (
              <div className="p-3.5 rounded-xl bg-rose-950/30 border border-rose-500/40 text-rose-300 text-xs flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold block">Capacity Warning: Utilization {capacityResult.utilizationPct}%</span>
                  <span>
                    Payload size ({formatBytes(payloadBytes)}) exceeds recommended safe carrier capacity ({formatBytes(capacityResult.recommendedSafeBytes)}). High likelihood of statistical steganalysis detection.
                  </span>
                </div>
              </div>
            )}

            {/* Embed Action Button */}
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
                  <Loader2 className="w-4 h-4 animate-spin text-white" />
                  <span>Executing Mock Embedding Pipeline...</span>
                </>
              ) : (
                <>
                  <Cpu className="w-4 h-4 text-brand-300" />
                  <span>Generate Stego Carrier Image</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Real Capacity Meter & Resulting Stego Preview */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <CapacityMeter capacity={capacityResult} showDetails={true} />

          {/* Stego Result Card */}
          {localStegoResult ? (
            <div className="p-6 rounded-2xl bg-slate-900/90 border border-emerald-500/30 shadow-2xl flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <h3 className="text-base font-bold text-white">Stego Image Generated</h3>
                </div>
                <SimulatedBadge />
              </div>

              {/* Stego Image Frame */}
              <div className="w-full h-56 rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center p-2">
                <img
                  src={localStegoResult.stegoImageDataUrl}
                  alt="Stego Result"
                  className="max-w-full max-h-full object-contain"
                />
              </div>

              {/* Real Metrics Summary */}
              {metrics && (
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">PSNR (Fidelity)</span>
                    <span className="font-mono font-bold text-emerald-300 text-sm">
                      {metrics.psnr} dB
                    </span>
                    <RealMathBadge size="sm" className="mt-1" />
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">SSIM (Similarity)</span>
                    <span className="font-mono font-bold text-brand-300 text-sm">
                      {metrics.ssim}
                    </span>
                    <RealMathBadge size="sm" className="mt-1" />
                  </div>
                </div>
              )}

              {/* Download Stego Button */}
              <button
                type="button"
                onClick={() => downloadDataUrl(localStegoResult.stegoImageDataUrl, `stego_${Date.now()}.png`)}
                className="w-full py-2.5 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs flex items-center justify-center gap-2 shadow-lg shadow-emerald-600/20 transition-all"
              >
                <Download className="w-4 h-4" />
                <span>Download Lossless Stego PNG</span>
              </button>

              {/* Exact-Image Secret Error Diff Canvas */}
              {activeMode === 'image' && (
                <div className="mt-2 pt-3 border-t border-slate-800">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold text-slate-300">Secret Image Error Diff</span>
                    <RealMathBadge label="Real Error Map" />
                  </div>
                  <div className="w-full h-36 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-center overflow-hidden">
                    <canvas ref={secretDiffCanvasRef} className="max-w-full max-h-full object-contain" />
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="p-8 rounded-2xl bg-slate-900/40 border border-dashed border-slate-800 text-center flex flex-col items-center justify-center min-h-[260px]">
              <Cpu className="w-8 h-8 text-slate-600 mb-2" />
              <p className="text-sm font-semibold text-slate-400">No Stego Image Generated Yet</p>
              <p className="text-xs text-slate-500 max-w-xs mt-1">
                Configure your cover and payload, enter an ephemeral passphrase, and click Generate to start.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
