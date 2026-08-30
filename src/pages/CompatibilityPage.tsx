import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDemo } from '../context/DemoContext';
import { ImageUploader } from '../components/common/ImageUploader';
import { CapacityMeter } from '../components/common/CapacityMeter';
import { SimulatedBadge } from '../components/common/SimulatedBadge';
import { RealMathBadge } from '../components/common/RealMathBadge';
import { VisualizationSuite } from '../components/visualizations/VisualizationSuite';
import { calculateCapacity, evaluateCompatibility } from '../utils/capacity';
import { estimateCompressedSize } from '../services/compression/compressionService';
import { estimateEncryptedSize } from '../services/crypto/cryptoService';
import { formatBytes } from '../utils/imageProcessing';
import {
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  Info,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  ArrowRight,
  FileText,
  Layers,
} from 'lucide-react';

export const CompatibilityPage: React.FC = () => {
  const navigate = useNavigate();
  const {
    coverDataUrl,
    coverMeta,
    coverTitle,
    payloadText,
    setCover,
    setPayloadText,
    setPayloadKind,
  } = useDemo();

  const [rawText, setRawText] = useState<string>(payloadText || '');
  const [textureScore, setTextureScore] = useState<number>(0.65);
  const [showWhy, setShowWhy] = useState<boolean>(true);
  const [showVisualizer, setShowVisualizer] = useState<boolean>(false);

  const [estCompressed, setEstCompressed] = useState<number>(0);
  const [estEncrypted, setEstEncrypted] = useState<number>(0);

  // Sync payload text with context
  const handleTextChange = (val: string) => {
    setRawText(val);
    setPayloadText(val);
  };

  // Real client-side UTF-8 byte calculation
  const rawBytes = useMemo(() => {
    return new TextEncoder().encode(rawText).length;
  }, [rawText]);

  const charCount = rawText.length;

  // Real capacity and compatibility evaluation
  const capacityResult = useMemo(() => {
    return calculateCapacity(coverMeta, rawBytes, textureScore);
  }, [coverMeta, rawBytes, textureScore]);

  const compatResult = useMemo(() => {
    return evaluateCompatibility(coverMeta, rawBytes, textureScore);
  }, [coverMeta, rawBytes, textureScore]);

  // Update mock compression and encryption size estimates
  useEffect(() => {
    let isMounted = true;
    if (rawBytes > 0) {
      Promise.all([
        estimateCompressedSize(rawBytes, 'text'),
        estimateEncryptedSize(rawBytes),
      ]).then(([comp, crypt]) => {
        if (isMounted) {
          setEstCompressed(comp.compressedBytes);
          setEstEncrypted(crypt.encryptedBytes);
        }
      });
    } else {
      setEstCompressed(0);
      setEstEncrypted(0);
    }
    return () => {
      isMounted = false;
    };
  }, [rawBytes]);

  const handleTextureComputed = (score: number) => {
    setTextureScore(score);
  };

  const getClassificationBadge = (cls: string) => {
    switch (cls) {
      case 'Excellent':
        return {
          bg: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
          icon: CheckCircle2,
        };
      case 'Good':
        return {
          bg: 'bg-teal-500/10 text-teal-300 border-teal-500/30',
          icon: CheckCircle2,
        };
      case 'Moderate':
        return {
          bg: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
          icon: AlertTriangle,
        };
      case 'Poor':
      case 'Unsupported':
        return {
          bg: 'bg-rose-500/10 text-rose-300 border-rose-500/30',
          icon: AlertOctagon,
        };
      default:
        return {
          bg: 'bg-slate-800 text-slate-300 border-slate-700',
          icon: Info,
        };
    }
  };

  const badgeStyle = getClassificationBadge(compatResult.classification);
  const StatusIcon = badgeStyle.icon;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-850 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <ShieldCheck className="w-6 h-6 text-brand-400" />
            <h1 className="text-2xl font-bold tracking-tight text-white">Cover Compatibility Analyzer</h1>
            <RealMathBadge label="Real Analysis" />
          </div>
          <p className="text-sm text-slate-400 max-w-3xl">
            Evaluate carrier image suitability, verify exact metadata, compute theoretical & safe capacity, and assess spatial texture masking resistance before embedding.
          </p>
        </div>

        {coverDataUrl && (
          <button
            type="button"
            onClick={() => {
              setPayloadKind('text');
              navigate('/embed');
            }}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-semibold text-sm shadow-lg shadow-brand-500/20 border border-brand-400/30 transition-all active:scale-95 self-start md:self-auto"
          >
            <span>Proceed to Embed</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Cover Carrier & Payload Input */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          {/* Cover Uploader */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800">
            <h2 className="text-base font-semibold text-white mb-3 flex items-center gap-2">
              <Layers className="w-4 h-4 text-brand-400" />
              <span>1. Select Carrier Image</span>
            </h2>

            <ImageUploader
              currentDataUrl={coverDataUrl}
              currentMeta={coverMeta}
              onImageSelected={(dataUrl, meta, title) => setCover(dataUrl, meta, title)}
              onClear={() => setCover('', null as any, '')}
            />
          </div>

          {/* Payload Configuration */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-base font-semibold text-white flex items-center gap-2">
                <FileText className="w-4 h-4 text-indigo-400" />
                <span>2. Secret Payload Definition</span>
              </h2>

              <div className="flex items-center gap-2">
                <RealMathBadge label="TextEncoder UTF-8" />
              </div>
            </div>

            <p className="text-xs text-slate-400 mb-3">
              Enter plaintext secret to inspect accurate multi-byte UTF-8 sizing, theoretical bounds, and compression estimates.
            </p>

            <textarea
              value={rawText}
              onChange={(e) => handleTextChange(e.target.value)}
              placeholder="Type or paste confidential payload text here..."
              rows={5}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3.5 text-sm text-slate-100 placeholder-slate-600 focus:border-brand-500 font-mono transition-colors"
            />

            {/* Live Size Breakdown */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 text-xs">
              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-850">
                <span className="text-slate-500 block text-[10px]">Characters</span>
                <span className="font-mono font-semibold text-slate-200 tabular-nums">
                  {charCount.toLocaleString()} chars
                </span>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-850">
                <div className="flex items-center justify-between">
                  <span className="text-slate-500 block text-[10px]">Raw UTF-8</span>
                  <span className="text-[9px] text-emerald-400 font-mono">Real</span>
                </div>
                <span className="font-mono font-semibold text-emerald-300 tabular-nums">
                  {formatBytes(rawBytes)}
                </span>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-850">
                <div className="flex items-center justify-between">
                  <span className="text-slate-500 block text-[10px]">Est. Compressed</span>
                  <SimulatedBadge size="sm" />
                </div>
                <span className="font-mono font-medium text-slate-300 tabular-nums">
                  {formatBytes(estCompressed)}
                </span>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-850">
                <div className="flex items-center justify-between">
                  <span className="text-slate-500 block text-[10px]">Est. Encrypted</span>
                  <SimulatedBadge size="sm" />
                </div>
                <span className="font-mono font-medium text-amber-300 tabular-nums">
                  {formatBytes(estEncrypted)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Compatibility Verdict, Capacity & "Why?" Math */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {/* Classification Banner Card */}
          <div className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Compatibility Rating
              </span>
              <div className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border ${badgeStyle.bg}`}>
                <StatusIcon className="w-4 h-4" />
                <span>{compatResult.classification.toUpperCase()}</span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-855">
              <p className="text-sm text-slate-200 leading-relaxed font-medium">
                {compatResult.reason}
              </p>
            </div>

            {/* Capacity Meter */}
            <CapacityMeter capacity={capacityResult} showDetails={true} />

            {/* Expandable "Why?" Breakdown */}
            <div className="border-t border-slate-800 pt-4">
              <button
                type="button"
                onClick={() => setShowWhy(!showWhy)}
                className="w-full flex items-center justify-between text-xs font-semibold text-slate-300 hover:text-white transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Info className="w-4 h-4 text-brand-400" />
                  <span>Why this rating? (Mathematical Derivation)</span>
                </div>
                {showWhy ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>

              {showWhy && (
                <div className="mt-3 p-3.5 rounded-xl bg-slate-950/90 border border-slate-850 text-xs text-slate-300 flex flex-col gap-2.5 font-mono">
                  <div className="flex items-center justify-between border-b border-slate-850 pb-1.5">
                    <span className="text-slate-400">Safe Headroom Ratio:</span>
                    <span className="font-bold text-brand-300">
                      {compatResult.headroomRatio}x
                    </span>
                  </div>

                  <div className="flex items-center justify-between border-b border-slate-850 pb-1.5">
                    <span className="text-slate-400">Texture Score:</span>
                    <span className="font-bold text-emerald-300">
                      {compatResult.textureScore.toFixed(2)} (top tercile ≥ 0.67)
                    </span>
                  </div>

                  <div className="flex items-center justify-between border-b border-slate-850 pb-1.5">
                    <span className="text-slate-400">Target Embedding bpp:</span>
                    <span className="font-bold text-slate-200">
                      {coverMeta ? ((rawBytes * 8) / (coverMeta.width * coverMeta.height * 3)).toFixed(4) : '0.0000'} bpp
                    </span>
                  </div>

                  <div className="text-[11px] text-slate-400 font-sans leading-normal pt-1">
                    Formula criteria:
                    <ul className="list-disc list-inside mt-1 space-y-0.5 text-slate-400">
                      <li><span className="text-emerald-300 font-mono">Excellent</span>: Headroom ≥ 4.0x & Texture ≥ 0.67</li>
                      <li><span className="text-teal-300 font-mono">Good</span>: Headroom ≥ 2.0x & Texture ≥ 0.40</li>
                      <li><span className="text-amber-300 font-mono">Moderate</span>: Headroom ≥ 1.0x (fits safe bounds)</li>
                      <li><span className="text-rose-300 font-mono">Poor</span>: Headroom &lt; 1.0x (exceeds safe capacity)</li>
                    </ul>
                  </div>
                </div>
              )}
            </div>

            {/* Quick Action to Visualizer */}
            {coverDataUrl && (
              <button
                type="button"
                onClick={() => setShowVisualizer(!showVisualizer)}
                className="w-full mt-2 py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center justify-center gap-2 border border-slate-700 transition-all"
              >
                <Layers className="w-4 h-4 text-brand-400" />
                <span>{showVisualizer ? 'Hide Spatial Texture Maps' : 'Inspect Edge & Texture Visualizations'}</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Spatial Texture Analysis Suite Component */}
      {coverDataUrl && (showVisualizer || true) && (
        <div className="mt-4 pt-6 border-t border-slate-800">
          <VisualizationSuite
            coverDataUrl={coverDataUrl}
            coverTitle={coverTitle}
            onTextureScoreComputed={handleTextureComputed}
          />
        </div>
      )}
    </div>
  );
};
