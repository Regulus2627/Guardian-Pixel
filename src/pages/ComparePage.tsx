import React, { useState, useEffect, useRef } from 'react';
import { useDemo } from '../context/DemoContext';
import { SplitSlider } from '../components/compare/SplitSlider';
import { MetricCard } from '../components/common/MetricCard';
import { RealMathBadge } from '../components/common/RealMathBadge';
import { extractImageData } from '../utils/imageProcessing';
import { computeImageMetrics, renderDifferenceCanvas } from '../utils/metrics';
import { assessDetectionRisk } from '../services/steganography/stegoService';
import { ImageQualityMetrics } from '../types';
import { downloadCanvas } from '../utils/download';
import {
  Sliders,
  Sparkles,
  Download,
  Eye,
} from 'lucide-react';

export const ComparePage: React.FC = () => {
  const { coverDataUrl, stegoResult, loadDemo } = useDemo();

  const [stegoUrl, setStegoUrl] = useState<string | null>(stegoResult?.stegoImageDataUrl || null);
  const [metrics, setMetrics] = useState<ImageQualityMetrics | null>(null);
  const [detectionRisk, setDetectionRisk] = useState<{ score: number; riskLevel: string } | null>(null);

  // Amplification controls
  const [ampFactor, setAmpFactor] = useState<number>(15);
  const [threshold, setThreshold] = useState<number>(0);
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.6);
  const [diffMode, setDiffMode] = useState<'diff' | 'heatmap' | 'overlay'>('diff');

  const diffCanvasRef = useRef<HTMLCanvasElement>(null);
  const rawCoverImgDataRef = useRef<ImageData | null>(null);
  const rawStegoImgDataRef = useRef<ImageData | null>(null);

  // Update stego image when context changes
  useEffect(() => {
    if (stegoResult?.stegoImageDataUrl) {
      setStegoUrl(stegoResult.stegoImageDataUrl);
    }
  }, [stegoResult]);

  // Compute real metrics & load pixel buffers
  useEffect(() => {
    let isMounted = true;
    if (coverDataUrl && stegoUrl) {
      Promise.all([extractImageData(coverDataUrl), extractImageData(stegoUrl)])
        .then(([cov, steg]) => {
          if (!isMounted) return;
          rawCoverImgDataRef.current = cov.imageData;
          rawStegoImgDataRef.current = steg.imageData;

          const payloadBytes = stegoResult?.payloadInfo?.rawBytes || 1024;
          const computed = computeImageMetrics(cov.imageData, steg.imageData, payloadBytes);
          setMetrics(computed);

          // Render canvas difference
          if (diffCanvasRef.current) {
            renderDifferenceCanvas(diffCanvasRef.current, cov.imageData, steg.imageData, {
              amplification: ampFactor,
              threshold,
              overlayOpacity,
              mode: diffMode,
            });
          }

          // Mock steganalysis detection score
          assessDetectionRisk(computed.bpp, 0.65).then((risk) => {
            if (isMounted) {
              setDetectionRisk({ score: risk.detectionScore, riskLevel: risk.riskLevel });
            }
          });
        })
        .catch((err) => {
          console.error(err);
        });
    }
    return () => {
      isMounted = false;
    };
  }, [coverDataUrl, stegoUrl, stegoResult]);

  // Re-render difference canvas when sliders change
  useEffect(() => {
    if (diffCanvasRef.current && rawCoverImgDataRef.current && rawStegoImgDataRef.current) {
      renderDifferenceCanvas(
        diffCanvasRef.current,
        rawCoverImgDataRef.current,
        rawStegoImgDataRef.current,
        {
          amplification: ampFactor,
          threshold,
          overlayOpacity,
          mode: diffMode,
        }
      );
    }
  }, [ampFactor, threshold, overlayOpacity, diffMode]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-850 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Sliders className="w-6 h-6 text-brand-400" />
            <h1 className="text-2xl font-bold tracking-tight text-white">Carrier & Stego Comparative Evaluation</h1>
            <RealMathBadge label="Real Math & Canvas" />
          </div>
          <p className="text-sm text-slate-400 max-w-3xl">
            Inspect per-pixel differential distortions, compute mathematical fidelity metrics (PSNR, SSIM, MSE), and analyze spatial embedding residuals.
          </p>
        </div>

        {!coverDataUrl && (
          <button
            type="button"
            onClick={() => loadDemo()}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-semibold text-sm shadow-md transition-all self-start md:self-auto"
          >
            <Sparkles className="w-4 h-4" />
            <span>Load Demo Session</span>
          </button>
        )}
      </div>

      {coverDataUrl && stegoUrl ? (
        <div className="flex flex-col gap-8">
          {/* Top Headline Metric Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <MetricCard
              label="PSNR (Fidelity)"
              value={metrics ? `${metrics.psnr}` : '--'}
              unit="dB"
              isSimulated={false}
              subtext="Theoretical limit: >40 dB imperceptible"
              tone="emerald"
            />

            <MetricCard
              label="SSIM (Structure)"
              value={metrics ? `${metrics.ssim}` : '--'}
              unit=""
              isSimulated={false}
              subtext="1.000 = identical perceptual structure"
              tone="emerald"
            />

            <MetricCard
              label="MSE (Mean Error)"
              value={metrics ? `${metrics.mse}` : '--'}
              unit=""
              isSimulated={false}
              subtext="Mean squared pixel error"
              tone="default"
            />

            <MetricCard
              label="Payload bpp"
              value={metrics ? `${metrics.bpp}` : '--'}
              unit="bpp"
              isSimulated={false}
              subtext="Bits embedded per channel pixel"
              tone="blue"
            />

            <MetricCard
              label="Steganalysis Risk"
              value={detectionRisk ? `${(detectionRisk.score * 100).toFixed(1)}%` : '14.2%'}
              unit={detectionRisk?.riskLevel || 'Low'}
              isSimulated={true}
              subtext="SRM Ensemble Classifier estimate"
              tone="amber"
            />

            <MetricCard
              label="Embed Latency"
              value={stegoResult ? `${stegoResult.embedTimeMs}` : '285'}
              unit="ms"
              isSimulated={true}
              subtext="Mock server compute time"
              tone="purple"
            />
          </div>

          {/* Interactive Split-View Slider */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-semibold text-white flex items-center gap-2">
                  <Eye className="w-4 h-4 text-brand-400" />
                  <span>Interactive Split Slider Comparison</span>
                </h2>
                <p className="text-xs text-slate-400">
                  Left side: Original cover carrier | Right side: Generated stego image
                </p>
              </div>
            </div>

            <SplitSlider
              originalSrc={coverDataUrl}
              modifiedSrc={stegoUrl}
              originalLabel="Original Cover"
              modifiedLabel="Stego Image"
            />
          </div>

          {/* Per-Pixel Difference Map & Amplification Visualizer */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-5">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-base font-semibold text-white">
                    Per-Pixel Difference Residual Map
                  </h2>
                  <RealMathBadge label="Direct Canvas Diff" />
                </div>
                <p className="text-xs text-slate-400 mt-0.5">
                  Calculates absolute delta |I₁(x,y,c) - I₂(x,y,c)| across all RGB color channels with dynamic amplification.
                </p>
              </div>

              {/* Diff Mode Selector */}
              <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
                <button
                  type="button"
                  onClick={() => setDiffMode('diff')}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                    diffMode === 'diff'
                      ? 'bg-slate-800 text-white shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Amplified Diff
                </button>
                <button
                  type="button"
                  onClick={() => setDiffMode('heatmap')}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                    diffMode === 'heatmap'
                      ? 'bg-slate-800 text-white shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Heatmap (Magma)
                </button>
                <button
                  type="button"
                  onClick={() => setDiffMode('overlay')}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                    diffMode === 'overlay'
                      ? 'bg-slate-800 text-white shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Cover Highlight
                </button>
              </div>
            </div>

            {/* Slider Controls */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-4 rounded-xl bg-slate-950/80 border border-slate-850">
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-300 font-medium">Amplification Factor:</span>
                  <span className="font-mono text-brand-400 font-bold">{ampFactor}×</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="50"
                  value={ampFactor}
                  onChange={(e) => setAmpFactor(parseInt(e.target.value))}
                  className="h-1.5 bg-slate-800 rounded-lg accent-brand-500 cursor-pointer"
                />
                <span className="text-[10px] text-slate-500">Magnifies 1-bit LSB subtle deviations for visual auditing</span>
              </div>

              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-300 font-medium">Threshold Filter:</span>
                  <span className="font-mono text-emerald-400 font-bold">{threshold} px</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="30"
                  value={threshold}
                  onChange={(e) => setThreshold(parseInt(e.target.value))}
                  className="h-1.5 bg-slate-800 rounded-lg accent-emerald-500 cursor-pointer"
                />
                <span className="text-[10px] text-slate-500">Filters out background noise under threshold magnitude</span>
              </div>

              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-300 font-medium">Overlay Opacity:</span>
                  <span className="font-mono text-amber-400 font-bold">{Math.round(overlayOpacity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.05"
                  value={overlayOpacity}
                  onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                  className="h-1.5 bg-slate-800 rounded-lg accent-amber-500 cursor-pointer"
                />
                <span className="text-[10px] text-slate-500">Alpha transparency when blending over cover image</span>
              </div>
            </div>

            {/* Difference Canvas Output */}
            <div className="relative w-full h-80 sm:h-[400px] rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-center p-2 overflow-hidden">
              <canvas ref={diffCanvasRef} className="max-w-full max-h-full object-contain" />

              <button
                type="button"
                onClick={() => {
                  if (diffCanvasRef.current) {
                    downloadCanvas(diffCanvasRef.current, `diff_map_${ampFactor}x_${Date.now()}.png`);
                  }
                }}
                className="absolute top-3 right-3 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/90 hover:bg-slate-800 text-slate-200 text-xs font-medium border border-slate-700 shadow-md backdrop-blur-sm transition-all"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Export Diff PNG</span>
              </button>
            </div>
          </div>

          {/* Detailed Channel Breakdown Table */}
          {metrics && (
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <h3 className="text-base font-semibold text-white">Channel-by-Channel Image Distortion Metrics</h3>
                <RealMathBadge label="Per-Channel Math" />
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950 border-b border-slate-800 text-slate-400 font-mono">
                    <tr>
                      <th className="p-3">Color Channel</th>
                      <th className="p-3">MSE (Mean Sq. Error)</th>
                      <th className="p-3">PSNR (Fidelity dB)</th>
                      <th className="p-3">Bit Plane Depth</th>
                      <th className="p-3">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    <tr className="hover:bg-slate-800/30">
                      <td className="p-3 flex items-center gap-2 font-semibold text-rose-400">
                        <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                        Red Channel (R)
                      </td>
                      <td className="p-3 text-slate-200">{metrics.channels.r.mse}</td>
                      <td className="p-3 text-emerald-400 font-semibold">{metrics.channels.r.psnr} dB</td>
                      <td className="p-3 text-slate-400">8-bit [0-255]</td>
                      <td className="p-3 text-emerald-300 font-sans">Lossless Carrier Bound</td>
                    </tr>
                    <tr className="hover:bg-slate-800/30">
                      <td className="p-3 flex items-center gap-2 font-semibold text-emerald-400">
                        <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                        Green Channel (G)
                      </td>
                      <td className="p-3 text-slate-200">{metrics.channels.g.mse}</td>
                      <td className="p-3 text-emerald-400 font-semibold">{metrics.channels.g.psnr} dB</td>
                      <td className="p-3 text-slate-400">8-bit [0-255]</td>
                      <td className="p-3 text-emerald-300 font-sans">Lossless Carrier Bound</td>
                    </tr>
                    <tr className="hover:bg-slate-800/30">
                      <td className="p-3 flex items-center gap-2 font-semibold text-sky-400">
                        <span className="w-2.5 h-2.5 rounded-full bg-sky-500"></span>
                        Blue Channel (B)
                      </td>
                      <td className="p-3 text-slate-200">{metrics.channels.b.mse}</td>
                      <td className="p-3 text-emerald-400 font-semibold">{metrics.channels.b.psnr} dB</td>
                      <td className="p-3 text-slate-400">8-bit [0-255]</td>
                      <td className="p-3 text-emerald-300 font-sans">Lossless Carrier Bound</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="p-12 rounded-2xl bg-slate-900/40 border border-dashed border-slate-800 text-center flex flex-col items-center justify-center min-h-[380px]">
          <Sliders className="w-10 h-10 text-slate-600 mb-3" />
          <h3 className="text-base font-bold text-slate-300">No Carrier vs Stego Pair Ready</h3>
          <p className="text-xs text-slate-500 max-w-sm mt-1 mb-4">
            Upload an image and run embedding in the Embed workflow, or click Load Demo to instantly populate a pre-calculated test session.
          </p>
          <button
            type="button"
            onClick={() => loadDemo()}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-semibold text-xs shadow-md transition-all"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Load Demo Session</span>
          </button>
        </div>
      )}
    </div>
  );
};
