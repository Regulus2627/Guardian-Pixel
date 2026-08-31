import React, { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  FileText,
  Loader2,
  ShieldCheck,
} from 'lucide-react';

import { ImageUploader } from '../components/common/ImageUploader';
import { useDemo } from '../context/DemoContext';
import {
  assessRealCompatibility,
  RealCompatibilityResult,
} from '../services/compatibility/compatibilityService';
import { formatBytes } from '../utils/imageProcessing';

function formatInteger(value: number): string {
  return Math.round(value).toLocaleString();
}

function formatBitsAsBytes(bits: number): string {
  return formatBytes(Math.ceil(bits / 8));
}

function classificationClasses(classification: string): string {
  switch (classification) {
    case 'Excellent':
      return 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300';
    case 'Good':
      return 'border-teal-500/40 bg-teal-500/10 text-teal-300';
    case 'Moderate':
      return 'border-amber-500/40 bg-amber-500/10 text-amber-300';
    default:
      return 'border-rose-500/40 bg-rose-500/10 text-rose-300';
  }
}

export const CompatibilityPage: React.FC = () => {
  const navigate = useNavigate();

  const {
    coverDataUrl,
    coverMeta,
    payloadText,
    setCover,
    setPayloadText,
    setPayloadKind,
  } = useDemo();

  const [text, setText] = useState<string>(payloadText || '');
  const [result, setResult] = useState<RealCompatibilityResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const characterCount = text.length;

  const utf8Bytes = useMemo(() => {
    return new TextEncoder().encode(text).length;
  }, [text]);

  const updateText = (value: string) => {
    setText(value);
    setPayloadText(value);
    setResult(null);
    setError(null);
  };

  const runCompatibility = async () => {
    if (!coverDataUrl) {
      setError('Please select a cover image.');
      return;
    }

    if (utf8Bytes <= 0) {
      setError('Please enter a secret message.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const response = await assessRealCompatibility({
        coverDataUrl,
        payloadKind: 'text',
        rawPayload: text,
      });

      setResult(response);
    } catch (caught: unknown) {
      setError(
        caught instanceof Error
          ? caught.message
          : 'Compatibility analysis failed.'
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const proceedToEmbed = () => {
    setPayloadKind('text');
    setPayloadText(text);
    navigate('/embed');
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-8">
      <div className="border-b border-slate-800 pb-6">
        <div className="flex items-center gap-2 mb-1">
          <ShieldCheck className="w-6 h-6 text-brand-400" />
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Real Cover–Payload Compatibility
          </h1>
          <span className="inline-flex items-center rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-300">
            Flask + HED
          </span>
        </div>

        <p className="text-sm text-slate-400 max-w-4xl">
          The backend analyses the real cover using HED, entropy and variance,
          calculates the encrypted payload requirement and reports whether the
          cover is recommended, risky or physically incompatible.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-300 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-semibold">Analysis failed</p>
            <p className="text-xs mt-1">{error}</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-7 flex flex-col gap-6">
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800">
            <h2 className="text-base font-semibold text-white mb-3">
              1. Select a Candidate Cover
            </h2>

            <ImageUploader
              currentDataUrl={coverDataUrl}
              currentMeta={coverMeta}
              onImageSelected={(dataUrl, meta, title) => {
                setCover(dataUrl, meta, title);
                setResult(null);
                setError(null);
              }}
              onClear={() => {
                setCover('', null as any, '');
                setResult(null);
              }}
            />
          </div>

          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800">
            <div className="flex items-center gap-2 mb-3">
              <FileText className="w-4 h-4 text-indigo-400" />
              <h2 className="text-base font-semibold text-white">
                2. Enter the Secret Text
              </h2>
            </div>

            <textarea
              value={text}
              onChange={(event) => updateText(event.target.value)}
              placeholder="Enter the text whose cover compatibility should be checked..."
              rows={7}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3.5 text-sm text-slate-100 font-mono focus:border-brand-500"
            />

            <div className="grid grid-cols-2 gap-3 mt-3">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <p className="text-[10px] uppercase text-slate-500">
                  Characters
                </p>
                <p className="font-mono font-semibold text-white">
                  {characterCount.toLocaleString()}
                </p>
              </div>

              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <p className="text-[10px] uppercase text-slate-500">
                  UTF-8 bytes
                </p>
                <p className="font-mono font-semibold text-white">
                  {formatBytes(utf8Bytes)}
                </p>
              </div>
            </div>

            <button
              type="button"
              disabled={isAnalyzing || !coverDataUrl || utf8Bytes === 0}
              onClick={runCompatibility}
              className={`w-full mt-4 py-3 rounded-xl text-sm font-bold flex items-center justify-center gap-2 ${
                isAnalyzing || !coverDataUrl || utf8Bytes === 0
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-brand-600 to-indigo-600 text-white hover:from-brand-500 hover:to-indigo-500'
              }`}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running real HED analysis...
                </>
              ) : (
                <>
                  <ShieldCheck className="w-4 h-4" />
                  Check Real Compatibility
                </>
              )}
            </button>
          </div>
        </div>

        <div className="lg:col-span-5">
          {result ? (
            <div className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 flex flex-col gap-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs text-slate-400">Compatibility Result</p>
                  <h2 className="text-xl font-bold text-white">
                    {result.cover.category} Cover
                  </h2>
                </div>

                <span
                  className={`px-3 py-1.5 rounded-full border text-sm font-bold ${classificationClasses(
                    result.classification
                  )}`}
                >
                  {result.classification}
                </span>
              </div>

              <div
                className={`p-4 rounded-xl border flex items-start gap-3 ${
                  result.recommended
                    ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200'
                    : 'border-amber-500/30 bg-amber-500/10 text-amber-200'
                }`}
              >
                {result.recommended ? (
                  <CheckCircle2 className="w-5 h-5 mt-0.5 flex-shrink-0" />
                ) : (
                  <AlertTriangle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                )}
                <div>
                  <p className="font-semibold">
                    {result.recommended
                      ? 'Recommended for embedding'
                      : result.compatible
                      ? 'Technically possible, but not recommended'
                      : 'Payload cannot fit in this cover'}
                  </p>
                  <p className="text-xs mt-1">{result.reason}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Texture score</p>
                  <p className="font-mono font-bold text-white">
                    {result.textureScore.toFixed(4)}
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Resolution</p>
                  <p className="font-mono font-bold text-white">
                    {result.cover.width} × {result.cover.height}
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Original payload</p>
                  <p className="font-mono font-bold text-white">
                    {formatBytes(result.payload.originalBytes)}
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Encrypted packet</p>
                  <p className="font-mono font-bold text-white">
                    {formatBitsAsBytes(result.payload.encryptedPacketBits)}
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Required with safety</p>
                  <p className="font-mono font-bold text-white">
                    {formatInteger(result.capacity.requiredWithSafety)} bits
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Maximum one-channel positions</p>
                  <p className="font-mono font-bold text-white">
                    {formatInteger(result.capacity.maximumPositions)} bits
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Remaining headroom</p>
                  <p className="font-mono font-bold text-white">
                    {formatInteger(result.capacity.remainingPositions)} bits
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Selected blocks</p>
                  <p className="font-mono font-bold text-white">
                    {result.locationMap.selectedBlocks.toLocaleString()} /{' '}
                    {result.locationMap.totalBlocks.toLocaleString()}
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Selected percentage</p>
                  <p className="font-mono font-bold text-white">
                    {result.locationMap.selectedPercentage.toFixed(2)}%
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <p className="text-slate-500">Map encoding</p>
                  <p className="font-mono font-bold text-white">
                    {result.locationMap.encoding} ({result.locationMap.encodedBytes} B)
                  </p>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-xs text-indigo-200">
                <p className="font-semibold">How this was calculated</p>
                <p className="mt-1">
                  Flask used real HED, entropy and variance, then included the
                  envelope, AES-GCM packet, 480 bootstrap bits, compressed map
                  metadata and the configured safety margin.
                </p>
                {result.isProvisional && (
                  <p className="mt-2 text-amber-300">
                    Recommendation bands are provisional until final baseline
                    and steganalysis evaluation is completed.
                  </p>
                )}
              </div>

              {result.recommended && (
                <button
                  type="button"
                  onClick={proceedToEmbed}
                  className="w-full py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-bold flex items-center justify-center gap-2"
                >
                  Proceed to Real Embedding
                  <ArrowRight className="w-4 h-4" />
                </button>
              )}
            </div>
          ) : (
            <div className="min-h-[440px] p-8 rounded-2xl bg-slate-900/40 border border-dashed border-slate-800 flex flex-col items-center justify-center text-center">
              <ShieldCheck className="w-10 h-10 text-slate-600 mb-3" />
              <p className="font-semibold text-slate-400">
                No real compatibility result yet
              </p>
              <p className="text-xs text-slate-500 max-w-sm mt-2">
                Select a cover, enter a payload and run the real backend
                analysis. The result will show the calculated cover class,
                encrypted requirement, available capacity and recommendation.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
