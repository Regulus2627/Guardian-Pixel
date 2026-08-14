import React, { useState, useEffect } from 'react';
import { TextureAnalysisResult } from '../../types';
import { analyzeCoverTexture } from '../../workers/workerClient';
import { RealMathBadge } from '../common/RealMathBadge';
import { OpacityOverlay } from './OpacityOverlay';
import { Download, Loader2, Layers, Eye, Cpu } from 'lucide-react';
import { downloadDataUrl } from '../../utils/download';

interface VisualizationSuiteProps {
  coverDataUrl: string;
  coverTitle?: string;
  onTextureScoreComputed?: (score: number, result: TextureAnalysisResult) => void;
}

type TabType = 'edge' | 'entropy' | 'variance' | 'fusion' | 'binary' | 'overlay';

export const VisualizationSuite: React.FC<VisualizationSuiteProps> = ({
  coverDataUrl,
  onTextureScoreComputed,
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('fusion');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<TextureAnalysisResult | null>(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    analyzeCoverTexture(coverDataUrl)
      .then((res) => {
        if (isMounted) {
          setAnalysisResult(res);
          setLoading(false);
          if (onTextureScoreComputed) {
            onTextureScoreComputed(res.textureScore, res);
          }
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Worker texture analysis failed');
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [coverDataUrl, onTextureScoreComputed]);

  if (loading) {
    return (
      <div className="p-8 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col items-center justify-center min-h-[360px] text-center">
        <Loader2 className="w-8 h-8 text-brand-400 animate-spin mb-3" />
        <h4 className="text-sm font-semibold text-white">Running Multi-threaded Spatial Texture Analysis</h4>
        <p className="text-xs text-slate-400 max-w-md mt-1">
          Processing Sobel gradients, local Shannon entropy, spatial variance windows, and Otsu binary segmentation in a background Web Worker...
        </p>
      </div>
    );
  }

  if (error || !analysisResult) {
    return (
      <div className="p-6 rounded-xl bg-rose-950/20 border border-rose-500/30 text-rose-300 text-sm">
        <p className="font-semibold">Analysis Failed</p>
        <p className="text-xs text-rose-400 mt-1">{error || 'Could not process image.'}</p>
      </div>
    );
  }

  const tabs: { id: TabType; name: string; icon: React.ComponentType<{ className?: string }> }[] = [
    { id: 'fusion', name: 'Multi-Feature Fusion', icon: Layers },
    { id: 'edge', name: 'Sobel Edge (HED Approx)', icon: Cpu },
    { id: 'entropy', name: 'Shannon Entropy', icon: Layers },
    { id: 'variance', name: 'Spatial Variance', icon: Layers },
    { id: 'binary', name: 'Carrier Segmentation', icon: Layers },
    { id: 'overlay', name: 'Interactive Overlay', icon: Eye },
  ];

  let currentMapUrl = analysisResult.fusionMapDataUrl;
  let currentTitle = 'Multi-Feature Fusion Map';
  let currentDesc = 'Normalized composite matrix integrating spatial variance (35%), Sobel edge gradients (35%), and localized Shannon entropy (30%).';
  let legendGradient = 'from-blue-700 via-amber-500 to-rose-600';
  let legendLabels = ['Low Texture', 'Moderate Texture', 'High Capacity Carrier'];

  if (activeTab === 'edge') {
    currentMapUrl = analysisResult.edgeMapDataUrl;
    currentTitle = 'Sobel Edge Map (Approximation)';
    currentDesc = 'Edge-detection approximation, not a full HED neural network. Highlights high-contrast spatial boundaries suitable for edge-adaptive steganography.';
    legendGradient = 'from-slate-950 via-cyan-900 to-cyan-400';
    legendLabels = ['Flat Smooth Surface', 'Moderate Gradient', 'Sharp High-Frequency Edge'];
  } else if (activeTab === 'entropy') {
    currentMapUrl = analysisResult.entropyMapDataUrl;
    currentTitle = 'Localized Shannon Entropy Map';
    currentDesc = 'Measures local information density and uncertainty across 7x7 pixel sliding neighborhoods. Higher entropy resists first-order histogram steganalysis.';
    legendGradient = 'from-slate-950 via-emerald-800 to-emerald-400';
    legendLabels = ['Low Uncertainty (0.0)', 'Medium Entropy (2.5)', 'High Entropy (4.0 bits)'];
  } else if (activeTab === 'variance') {
    currentMapUrl = analysisResult.varianceMapDataUrl;
    currentTitle = 'Local Spatial Variance Map';
    currentDesc = 'Second-order statistical dispersion across 5x5 pixel windows. High-variance regions naturally mask LSB flipping from human visual inspection.';
    legendGradient = 'from-slate-950 via-amber-800 to-amber-400';
    legendLabels = ['Homogeneous / Zero Variance', 'Intermediate Variance', 'Extreme High Variance'];
  } else if (activeTab === 'binary') {
    currentMapUrl = analysisResult.binaryMapDataUrl;
    currentTitle = 'Binary Carrier Segmentation Mask';
    currentDesc = 'Otsu-derived threshold segmentation isolating optimal high-capacity carrier zones (cyan) from vulnerable smooth surfaces (slate).';
    legendGradient = 'from-slate-900 to-sky-400';
    legendLabels = ['Vulnerable Smooth Region (Reject)', 'Optimal Carrier Zone (Embed)'];
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Header with Stats Summary */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 p-4 rounded-xl bg-slate-900/90 border border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-white">Spatial Texture & Edge Analysis Suite</h3>
            <RealMathBadge label="Web Worker Real Math" />
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Client-side image analysis computed at {analysisResult.width}×{analysisResult.height}px resolution
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs">
          <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
            <span className="text-slate-400 block text-[10px]">Texture Score</span>
            <span className="font-mono font-bold text-brand-300 text-sm">
              {analysisResult.textureScore.toFixed(2)} / 1.00
            </span>
          </div>

          <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
            <span className="text-slate-400 block text-[10px]">Mean Entropy</span>
            <span className="font-mono font-medium text-emerald-300 text-sm">
              {analysisResult.meanEntropy.toFixed(2)} bits
            </span>
          </div>

          <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
            <span className="text-slate-400 block text-[10px]">Mean Variance</span>
            <span className="font-mono font-medium text-amber-300 text-sm">
              {analysisResult.meanVariance.toFixed(1)} σ²
            </span>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 border-b border-slate-800">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;

          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-3 py-2 text-xs font-medium rounded-lg whitespace-nowrap transition-all ${
                isActive
                  ? 'bg-slate-800 text-white border border-slate-700 font-semibold'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-brand-400' : 'text-slate-500'}`} />
              <span>{tab.name}</span>
            </button>
          );
        })}
      </div>

      {/* Main Map Viewer or Overlay View */}
      {activeTab === 'overlay' ? (
        <OpacityOverlay
          originalDataUrl={coverDataUrl}
          overlayDataUrl={analysisResult.fusionMapDataUrl}
          overlayTitle="Texture Fusion Mask"
        />
      ) : (
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col gap-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h4 className="text-sm font-semibold text-white">{currentTitle}</h4>
              <p className="text-xs text-slate-400 mt-0.5">{currentDesc}</p>
            </div>

            <button
              type="button"
              onClick={() => downloadDataUrl(currentMapUrl, `${activeTab}_map_${Date.now()}.png`)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all self-start sm:self-auto"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export PNG</span>
            </button>
          </div>

          {/* Map Preview Frame */}
          <div className="w-full h-80 sm:h-96 rounded-lg overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center p-2 relative">
            <img
              src={currentMapUrl}
              alt={currentTitle}
              className="max-w-full max-h-full object-contain"
            />
          </div>

          {/* Color Legend Bar */}
          <div className="mt-1 pt-3 border-t border-slate-800/80">
            <div className={`h-2.5 w-full rounded-full bg-gradient-to-r ${legendGradient} mb-1.5`} />
            <div className="flex items-center justify-between text-[10px] font-mono text-slate-400">
              {legendLabels.map((lbl, idx) => (
                <span key={idx}>{lbl}</span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
