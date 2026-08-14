import React, { useState, useRef, useCallback, useEffect } from 'react';
import { SlidersHorizontal } from 'lucide-react';

interface SplitSliderProps {
  originalSrc: string;
  modifiedSrc: string;
  originalLabel?: string;
  modifiedLabel?: string;
}

export const SplitSlider: React.FC<SplitSliderProps> = ({
  originalSrc,
  modifiedSrc,
  originalLabel = 'Original Cover',
  modifiedLabel = 'Stego Carrier',
}) => {
  const [sliderPos, setSliderPos] = useState<number>(50); // percentage 0-100
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(pct);
  }, []);

  const handleMouseDown = () => setIsDragging(true);
  const handleTouchStart = () => setIsDragging(true);

  useEffect(() => {
    const handleMouseUp = () => setIsDragging(false);
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) handleMove(e.clientX);
    };
    const handleTouchMove = (e: TouchEvent) => {
      if (isDragging && e.touches[0]) handleMove(e.touches[0].clientX);
    };

    window.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('touchend', handleMouseUp);
    window.addEventListener('touchmove', handleTouchMove);

    return () => {
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('touchend', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
    };
  }, [isDragging, handleMove]);

  return (
    <div className="flex flex-col gap-2">
      <div
        ref={containerRef}
        className="relative w-full h-80 sm:h-[420px] rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 select-none cursor-ew-resize group"
        onMouseDown={handleMouseDown}
        onTouchStart={handleTouchStart}
      >
        {/* Under Layer: Stego Carrier (Right Side) */}
        <img
          src={modifiedSrc}
          alt={modifiedLabel}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />

        {/* Top Layer: Original Cover (Left Side, clipped to sliderPos) */}
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ clipPath: `polygon(0 0, ${sliderPos}% 0, ${sliderPos}% 100%, 0 100%)` }}
        >
          <img
            src={originalSrc}
            alt={originalLabel}
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          />
        </div>

        {/* Divider Handle Line */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-brand-400 shadow-2xl z-20"
          style={{ left: `${sliderPos}%` }}
        >
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-slate-900 border-2 border-brand-400 flex items-center justify-center text-brand-300 shadow-xl group-hover:scale-110 transition-transform">
            <SlidersHorizontal className="w-4 h-4" />
          </div>
        </div>

        {/* Bottom Floating Badges */}
        <div className="absolute bottom-3 left-3 z-10 bg-slate-950/85 backdrop-blur-sm px-2.5 py-1 rounded text-[11px] font-mono text-slate-300 border border-slate-800">
          ◀ {originalLabel}
        </div>
        <div className="absolute bottom-3 right-3 z-10 bg-slate-950/85 backdrop-blur-sm px-2.5 py-1 rounded text-[11px] font-mono text-brand-300 border border-brand-500/30">
          {modifiedLabel} ▶
        </div>
      </div>

      <div className="flex items-center justify-between text-xs text-slate-400 px-1">
        <span>Drag center divider horizontally to compare fine pixel structures</span>
        <span className="font-mono text-brand-400 font-semibold">{Math.round(sliderPos)}% / {Math.round(100 - sliderPos)}%</span>
      </div>
    </div>
  );
};
