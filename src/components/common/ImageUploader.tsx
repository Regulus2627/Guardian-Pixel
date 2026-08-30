import React, { useRef, useState } from 'react';
import { UploadCloud, CheckCircle, Sparkles, X } from 'lucide-react';
import { CoverImageMeta, SampleImageItem } from '../../types';
import { extractCoverMetadata, extractDataUrlMetadata, formatBytes, readFileAsDataUrl } from '../../utils/imageProcessing';
import { getSampleCoverImages } from '../../utils/sampleData';

interface ImageUploaderProps {
  label?: string;
  description?: string;
  currentDataUrl: string | null;
  currentMeta?: CoverImageMeta | null;
  onImageSelected: (dataUrl: string, meta: CoverImageMeta, title: string) => void;
  onClear?: () => void;
  showSamplePresets?: boolean;
  minDimension?: number;
  className?: string;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  label = 'Select Cover Carrier Image',
  description = 'Supports PNG, JPEG, WEBP, BMP (lossless PNG strongly recommended)',
  currentDataUrl,
  currentMeta,
  onImageSelected,
  onClear,
  showSamplePresets = true,
  minDimension = 64,
  className = '',
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const sampleCovers = getSampleCoverImages();

  const handleFile = async (file: File) => {
    setErrorMsg(null);
    if (!file.type.startsWith('image/')) {
      setErrorMsg('Invalid file format. Please upload an image file (PNG, JPG, WEBP, BMP).');
      return;
    }

    try {
      const dataUrl = await readFileAsDataUrl(file);
      const meta = await extractCoverMetadata(file, dataUrl);

      if (meta.width < minDimension || meta.height < minDimension) {
        setErrorMsg(`Image resolution (${meta.width}x${meta.height}) is too small. Minimum required is ${minDimension}x${minDimension}px.`);
        return;
      }

      onImageSelected(dataUrl, meta, file.name);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to process uploaded image.');
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleSelectSample = async (sample: SampleImageItem) => {
    setErrorMsg(null);
    try {
      const meta = await extractDataUrlMetadata(sample.dataUrl, sample.title);
      onImageSelected(sample.dataUrl, meta, sample.title);
    } catch (err: any) {
      setErrorMsg('Failed to load sample image.');
    }
  };

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-semibold text-slate-200 block">{label}</label>
          <p className="text-xs text-slate-400">{description}</p>
        </div>

        {currentDataUrl && onClear && (
          <button
            onClick={onClear}
            type="button"
            className="inline-flex items-center gap-1 text-xs text-slate-400 hover:text-rose-400 transition-colors"
          >
            <X className="w-3.5 h-3.5" />
            <span>Remove</span>
          </button>
        )}
      </div>

      {errorMsg && (
        <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center justify-between">
          <span>{errorMsg}</span>
          <button onClick={() => setErrorMsg(null)} className="text-rose-400 hover:text-white">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {currentDataUrl ? (
        <div className="relative rounded-xl border border-slate-700 bg-slate-900/90 overflow-hidden group">
          <div className="flex flex-col sm:flex-row items-center gap-4 p-4">
            <div className="relative w-40 h-32 flex-shrink-0 rounded-lg overflow-hidden border border-slate-800 bg-slate-950 flex items-center justify-center">
              <img
                src={currentDataUrl}
                alt="Selected Carrier"
                className="w-full h-full object-contain"
              />
            </div>

            <div className="flex-1 min-w-0 flex flex-col justify-between py-1">
              <div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                  <span className="font-semibold text-sm text-white truncate">Image Loaded & Validated</span>
                </div>

                {currentMeta && (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 text-xs">
                    <div className="bg-slate-950/60 p-2 rounded border border-slate-800/80">
                      <span className="text-slate-400 block text-[10px]">Dimensions</span>
                      <span className="font-mono font-medium text-slate-200">
                        {currentMeta.width} × {currentMeta.height} px
                      </span>
                    </div>
                    <div className="bg-slate-950/60 p-2 rounded border border-slate-800/80">
                      <span className="text-slate-400 block text-[10px]">Format</span>
                      <span className="font-mono font-medium text-slate-200">{currentMeta.fileType}</span>
                    </div>
                    <div className="bg-slate-950/60 p-2 rounded border border-slate-800/80">
                      <span className="text-slate-400 block text-[10px]">File Size</span>
                      <span className="font-mono font-medium text-slate-200">
                        {formatBytes(currentMeta.fileSizeBytes)}
                      </span>
                    </div>
                    <div className="bg-slate-950/60 p-2 rounded border border-slate-800/80">
                      <span className="text-slate-400 block text-[10px]">Channels / Depth</span>
                      <span className="font-mono font-medium text-slate-200">
                        {currentMeta.channels}ch / {currentMeta.bitDepth || 8}bpc
                      </span>
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-3 flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="text-xs text-brand-400 hover:text-brand-300 underline font-medium"
                >
                  Change image
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
            isDragging
              ? 'border-brand-400 bg-brand-500/10'
              : 'border-slate-700 hover:border-slate-500 bg-slate-900/40 hover:bg-slate-900/70'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/bmp"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFile(e.target.files[0]);
              }
            }}
          />

          <div className="w-12 h-12 mx-auto rounded-full bg-slate-800/80 flex items-center justify-center text-brand-400 mb-3 group-hover:scale-110 transition-transform">
            <UploadCloud className="w-6 h-6" />
          </div>

          <p className="text-sm font-medium text-slate-200">
            Click to upload or drag & drop carrier image
          </p>
          <p className="text-xs text-slate-500 mt-1">PNG, JPG, WEBP or BMP (lossless format recommended)</p>
        </div>
      )}

      {/* Preset Sample Gallery */}
      {showSamplePresets && (
        <div className="mt-1">
          <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-2">
            <Sparkles className="w-3.5 h-3.5 text-brand-400" />
            <span>Or select a calibrated research sample:</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
            {sampleCovers.map((sample) => (
              <button
                key={sample.id}
                type="button"
                onClick={() => handleSelectSample(sample)}
                className="flex flex-col text-left p-2 rounded-lg bg-slate-900/70 hover:bg-slate-800 border border-slate-800 hover:border-slate-600 transition-all text-xs group"
              >
                <div className="w-full h-14 rounded overflow-hidden mb-1.5 bg-slate-950 relative border border-slate-800">
                  <img
                    src={sample.dataUrl}
                    alt={sample.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                  />
                  <span className="absolute bottom-1 right-1 text-[9px] px-1 py-0.2 rounded bg-slate-950/80 text-slate-300 font-mono">
                    {sample.expectedTexture}
                  </span>
                </div>
                <span className="font-medium text-slate-200 truncate group-hover:text-brand-300">
                  {sample.title}
                </span>
                <span className="text-[10px] text-slate-500">{sample.category}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
