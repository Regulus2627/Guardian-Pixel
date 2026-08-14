import React from 'react';
import {
  BookOpen,
  ShieldAlert,
  Calculator,
  Layers,
  Sparkles,
  Binary,
} from 'lucide-react';
import { SimulatedBadge } from '../components/common/SimulatedBadge';
import { RealMathBadge } from '../components/common/RealMathBadge';

export const MethodologyPage: React.FC = () => {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-10">
      {/* Header */}
      <div className="border-b border-slate-850 pb-6">
        <div className="flex items-center gap-2 mb-1">
          <BookOpen className="w-6 h-6 text-brand-400" />
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Scientific Methodology & Mathematical Foundations
          </h1>
        </div>
        <p className="text-sm text-slate-400 max-w-3xl">
          Theoretical architecture, algorithmic pipelines, quality metrics, carrier threat models, and simulation disclosures for the StegoLab platform.
        </p>
      </div>

      {/* Prominent Mock Disclosure Banner */}
      <div className="p-5 rounded-2xl bg-slate-900 border border-brand-500/30 text-slate-200 flex flex-col sm:flex-row items-start gap-4">
        <div className="w-10 h-10 rounded-xl bg-brand-500/20 text-brand-400 flex items-center justify-center flex-shrink-0">
          <Sparkles className="w-5 h-5" />
        </div>
        <div className="text-xs space-y-1">
          <div className="flex items-center gap-2">
            <span className="font-bold text-sm text-white">System Architecture & Mock Disclosures</span>
            <SimulatedBadge label="Frontend Workbench" />
          </div>
          <p className="text-slate-300 leading-relaxed">
            In this demonstration build, deep neural carrier models, real hardware-accelerated cryptographic primitives, and steganalytic classifiers are structured behind clean client wrappers (<code className="text-brand-300">services/apiClient/</code>) with deterministic simulations.
          </p>
          <p className="text-slate-400 leading-relaxed">
            All <strong>image metadata extraction, UTF-8 byte counting, capacity arithmetic, PSNR, SSIM, MSE, differential pixel canvases, local Shannon entropy, spatial variance, and Sobel edge detection</strong> are computed genuinely in real-time in the browser client and Web Worker threads.
          </p>
        </div>
      </div>

      {/* Section 1: Algorithmic Pipelines */}
      <div className="flex flex-col gap-6">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Layers className="w-5 h-5 text-brand-400" />
          <span>1. End-to-End Steganographic Architecture</span>
        </h2>

        {/* Embedding Pipeline Stepper */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-4">
          <h3 className="text-sm font-semibold text-white">Embedding Pipeline (Forward Flow)</h3>
          <div className="grid grid-cols-1 sm:grid-cols-7 gap-2 text-xs">
            {[
              { step: '1', title: 'Input Carrier', desc: 'Load & validate cover dimensions & color space' },
              { step: '2', title: 'Compatibility', desc: 'Spatial texture & capacity headroom verification' },
              { step: '3', title: 'Payload Prep', desc: 'UTF-8 encoding / binary packaging' },
              { step: '4', title: 'Compression', desc: 'Zlib RFC-1950 entropy pre-pass' },
              { step: '5', title: 'Encryption', desc: 'AES-256-GCM + 96-bit IV + HMAC Tag' },
              { step: '6', title: 'Embedding', desc: 'Adaptive variance-guided LSB modification' },
              { step: '7', title: 'Stego Output', desc: 'Lossless PNG carrier generation' },
            ].map((st) => (
              <div key={st.step} className="p-3 rounded-xl bg-slate-950 border border-slate-855 flex flex-col gap-1">
                <span className="w-5 h-5 rounded-full bg-brand-500/20 text-brand-300 border border-brand-500/30 flex items-center justify-center font-mono font-bold text-[10px]">
                  {st.step}
                </span>
                <span className="font-semibold text-slate-200 text-[11px]">{st.title}</span>
                <span className="text-[10px] text-slate-500 leading-tight">{st.desc}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Extraction Pipeline Stepper */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-4">
          <h3 className="text-sm font-semibold text-white">Extraction Pipeline (Reverse Flow)</h3>
          <div className="grid grid-cols-1 sm:grid-cols-6 gap-2 text-xs">
            {[
              { step: '1', title: 'Carrier Ingest', desc: 'Verify container PNG integrity & bit depth' },
              { step: '2', title: 'Header Parse', desc: 'Scan steganographic envelope & magic bytes' },
              { step: '3', title: 'Auth & Key', desc: 'Derive symmetric key from passphrase' },
              { step: '4', title: 'Bit Extraction', desc: 'Sample payload bits from adaptive coordinates' },
              { step: '5', title: 'Decryption', desc: 'Verify HMAC auth tag & decrypt ciphertext' },
              { step: '6', title: 'Decompress', desc: 'Restore exact plaintext / binary stream' },
            ].map((st) => (
              <div key={st.step} className="p-3 rounded-xl bg-slate-950 border border-slate-850 flex flex-col gap-1">
                <span className="w-5 h-5 rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/30 flex items-center justify-center font-mono font-bold text-[10px]">
                  {st.step}
                </span>
                <span className="font-semibold text-slate-200 text-[11px]">{st.title}</span>
                <span className="text-[10px] text-slate-500 leading-tight">{st.desc}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Section 2: Exact-Image Mode vs. Approximate-Autoencoder Mode */}
      <div className="flex flex-col gap-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Binary className="w-5 h-5 text-indigo-400" />
          <span>2. Exact-Image Mode vs. Approximate Autoencoder Mode</span>
        </h2>

        <p className="text-xs text-slate-300 leading-relaxed">
          Modern steganographic paradigms bifurcate into <strong>exact discrete embedding</strong> (where every payload bit is recovered perfectly without single-bit error) and <strong>approximate deep neural autoencoding</strong> (where secret imagery is encoded into latent representations and reconstructed perceptually via neural decoders).
        </p>

        <div className="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-900/60 shadow-lg">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950 border-b border-slate-800 text-slate-300 font-mono">
              <tr>
                <th className="p-3.5 w-1/4">Property</th>
                <th className="p-3.5 w-3/8 text-brand-300">Exact-image mode</th>
                <th className="p-3.5 w-3/8 text-teal-300">Approximate autoencoder mode</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70 text-slate-300">
              <tr className="hover:bg-slate-800/30">
                <td className="p-3.5 font-semibold text-white">Secret recovery</td>
                <td className="p-3.5 font-mono text-emerald-400">Exact</td>
                <td className="p-3.5 font-mono text-amber-300">Approximate</td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="p-3.5 font-semibold text-white">Pixel-level equality</td>
                <td className="p-3.5">Required/expected</td>
                <td className="p-3.5">Not guaranteed</td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="p-3.5 font-semibold text-white">Reconstruction</td>
                <td className="p-3.5">Deterministic where supported</td>
                <td className="p-3.5">Learned reconstruction</td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="p-3.5 font-semibold text-white">Evaluation</td>
                <td className="p-3.5">Pixel-perfect metrics possible (MSE = 0, PSNR = ∞)</td>
                <td className="p-3.5">Perceptual metrics important (LPIPS, DISTS, SSIM)</td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="p-3.5 font-semibold text-white">Capacity</td>
                <td className="p-3.5">Potentially higher requirements</td>
                <td className="p-3.5">Model-dependent latent compression</td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="p-3.5 font-semibold text-white">Failure mode</td>
                <td className="p-3.5 text-rose-300">Extraction/integrity failure (all-or-nothing)</td>
                <td className="p-3.5 text-amber-300">Reconstruction degradation (blur/hallucination)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Section 3: Mathematical Metric Definitions */}
      <div className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Calculator className="w-5 h-5 text-emerald-400" />
            <span>3. Mathematical Quality & Fidelity Formulations</span>
          </h2>
          <RealMathBadge label="Implemented in JavaScript" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* MSE Formula */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-2">
            <span className="font-bold text-sm text-white">Mean Squared Error (MSE)</span>
            <p className="text-xs text-slate-400">
              Quantifies cumulative squared difference between original carrier I₁ and stego image I₂ across width W, height H, and color channels C.
            </p>
            <pre className="p-3 rounded-xl bg-slate-950 border border-slate-850 text-xs font-mono text-emerald-300 overflow-x-auto">
{`MSE = (1 / (3 * W * H)) * Σ [ I₁(x,y,c) - I₂(x,y,c) ]²`}
            </pre>
          </div>

          {/* PSNR Formula */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-2">
            <span className="font-bold text-sm text-white">Peak Signal-to-Noise Ratio (PSNR)</span>
            <p className="text-xs text-slate-400">
              Logarithmic ratio between peak signal power (255²) and mean squared error distortion in decibels (dB).
            </p>
            <pre className="p-3 rounded-xl bg-slate-950 border border-slate-850 text-xs font-mono text-brand-300 overflow-x-auto">
{`PSNR = 10 * log₁₀( 255² / MSE )   [dB]`}
            </pre>
          </div>

          {/* SSIM Formula */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-2">
            <span className="font-bold text-sm text-white">Structural Similarity Index (SSIM)</span>
            <p className="text-xs text-slate-400">
              Windowed perceptual quality metric combining luminance (μ), contrast (σ), and cross-covariance (σ_xy) with stability constants C₁ = 6.5025, C₂ = 58.5225.
            </p>
            <pre className="p-3 rounded-xl bg-slate-950 border border-slate-850 text-xs font-mono text-indigo-300 overflow-x-auto">
{`SSIM(x,y) = ( (2μ_x μ_y + C₁)(2σ_xy + C₂) ) /
            ( (μ_x² + μ_y² + C₁)(σ_x² + σ_y² + C₂) )`}
            </pre>
          </div>

          {/* Shannon Entropy */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-2">
            <span className="font-bold text-sm text-white">Localized Shannon Entropy</span>
            <p className="text-xs text-slate-400">
              Measures uncertainty and information richness over N×N neighborhood intensity probability distribution p_i.
            </p>
            <pre className="p-3 rounded-xl bg-slate-950 border border-slate-850 text-xs font-mono text-amber-300 overflow-x-auto">
{`H(X) = - Σ [ p_i * log₂(p_i) ]   [bits/pixel]`}
            </pre>
          </div>
        </div>
      </div>

      {/* Section 4: Steganographic Threat Model */}
      <div className="flex flex-col gap-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-rose-400" />
          <span>4. Comprehensive Steganographic Threat Model</span>
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col gap-2">
            <span className="font-bold text-rose-300">1. Statistical Steganalysis (SRM)</span>
            <p className="text-slate-400 leading-relaxed">
              Spatial Rich Model (SRM) extracts 39-million sub-model high-order co-occurrence matrices from noise residuals to identify non-natural pixel statistics.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col gap-2">
            <span className="font-bold text-amber-300">2. Human Visual Inspection</span>
            <p className="text-slate-400 leading-relaxed">
              Detection of contour artifacts, unnatural color noise, or bit-plane pattern anomalies across smooth low-texture surfaces.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col gap-2">
            <span className="font-bold text-brand-300">3. Active Extraction & Key Guessing</span>
            <p className="text-slate-400 leading-relaxed">
              Adversary attempts to derive key parameters through dictionary attacks, mitigated by PBKDF2/Argon2 key stretching and AES-GCM MAC validation.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col gap-2">
            <span className="font-bold text-purple-300">4. Lossy JPEG Recompression</span>
            <p className="text-slate-400 leading-relaxed">
              DCT quantization destroys spatial LSB planes. Spatial steganography requires lossless transmission (PNG, WEBP lossless, BMP).
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col gap-2">
            <span className="font-bold text-teal-300">5. Geometric Distortions</span>
            <p className="text-slate-400 leading-relaxed">
              Rescaling, cropping, or rotation desynchronizes spatial embedding coordinates, causing catastrophic extraction failure.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col gap-2">
            <span className="font-bold text-emerald-300">6. Carrier Format Conversion</span>
            <p className="text-slate-400 leading-relaxed">
              Transcoding to lossy or altered color profile spaces will alter least significant bit planes, invalidating ciphertext envelopes.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
