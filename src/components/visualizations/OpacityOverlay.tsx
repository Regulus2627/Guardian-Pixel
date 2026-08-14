import React, { useState } from 'react';
import { Eye } from 'lucide-react';

interface OpacityOverlayProps {
  originalDataUrl: string;
  overlayDataUrl: string;
  overlayTitle: string;
}

export const OpacityOverlay: React.FC<OpacityOverlayProps> = ({
  originalDataUrl,
  overlayDataUrl,
  overlayTitle,
}) => {
  const [opacity, setOpacity] = useState<number>(0.65);
  const [blendMode, setBlendMode] = useState<'normal' | 'screen' | 'multiply' | 'color-dodge'>('normal');

  return (
    <div className="flex flex-col gap-3 p-4 rounded-xl bg-slate-900/90 border border-slate-800">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-brand-400" />
          <h4 className="text-sm font-semibold text-white">Interactive Composite Overlay</h4>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Blend:</span>
            <select
              value={blendMode}
              onChange={(e) => setBlendMode(e.target.value as any)}
              className="text-xs bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200"
            >
              <option value="normal">Normal</option>
              <option value="screen">Screen (Lighten)</option>
              <option value="multiply">Multiply (Darken)</option>
              <option value="color-dodge">Color Dodge</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Opacity: {Math.round(opacity * 100)}%</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-24 h-1.5 bg-slate-800 rounded-lg accent-brand-500 cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* Layered Canvas Display */}
      <div className="relative w-full h-80 sm:h-96 rounded-lg overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center">
        {/* Base Layer: Original Image */}
        <img
          src={originalDataUrl}
          alt="Original Cover"
          className="absolute max-w-full max-h-full object-contain pointer-events-none"
        />

        {/* Overlay Layer: Feature Map */}
        <img
          src={overlayDataUrl}
          alt={overlayTitle}
          className="absolute max-w-full max-h-full object-contain pointer-events-none transition-opacity duration-75"
          style={{
            opacity,
            mixBlendMode: blendMode,
          }}
        />

        <div className="absolute bottom-2 left-2 bg-slate-950/80 backdrop-blur-sm px-2.5 py-1 rounded text-[11px] font-mono text-slate-300 border border-slate-800">
          Cover + {overlayTitle} ({Math.round(opacity * 100)}% opacity)
        </div>
      </div>
    </div>
  );
};
