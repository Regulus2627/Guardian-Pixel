import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useDemo } from '../context/DemoContext';
import {
  ShieldCheck,
  Cpu,
  Sliders,
  BarChart3,
  Sparkles,
  ArrowRight,
  Binary,
  Lock,
} from 'lucide-react';
import { SimulatedBadge } from '../components/common/SimulatedBadge';
import { RealMathBadge } from '../components/common/RealMathBadge';

export const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const { loadDemo } = useDemo();

  const handleStartDemo = async () => {
    await loadDemo();
    navigate('/compatibility');
  };

  const workflowSteps = [
    {
      step: '01',
      title: 'Cover Carrier Selection',
      desc: 'Select or upload an image and extract real pixel dimensions, format, and color channel parameters.',
      path: '/compatibility',
    },
    {
      step: '02',
      title: 'Compatibility Analysis',
      desc: 'Evaluate spatial entropy, local variance, and safe capacity headroom bounds with live mathematical formulas.',
      path: '/compatibility',
    },
    {
      step: '03',
      title: 'Payload Preparation',
      desc: 'Define plaintext, files, or secret imagery with real UTF-8 sizing, mock encryption, and compression.',
      path: '/embed',
    },
    {
      step: '04',
      title: 'Stego Generation',
      desc: 'Generate lossless carrier PNGs with adaptive LSB masking and realistic pipeline latency.',
      path: '/embed',
    },
    {
      step: '05',
      title: 'Extraction & Auth',
      desc: 'Recover embedded secrets with passphrase-authenticated decryption and integrity validation.',
      path: '/extract',
    },
    {
      step: '06',
      title: 'Comparative Evaluation',
      desc: 'Inspect real PSNR, SSIM, MSE metrics, split-slider views, and amplified difference canvases.',
      path: '/compare',
    },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 flex flex-col gap-12">
      {/* Hero Section */}
      <div className="relative rounded-3xl bg-gradient-to-b from-slate-900/90 via-slate-900/50 to-slate-950/80 border border-slate-800 p-8 sm:p-12 overflow-hidden shadow-2xl">
        {/* Glow ambient background effect */}
        <div className="absolute top-0 right-1/4 w-96 h-96 bg-brand-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl flex flex-col gap-6">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-brand-500/10 text-brand-300 border border-brand-500/30 text-xs font-semibold">
              <Binary className="w-3.5 h-3.5" />
              Scientific Carrier Evaluation
            </span>
            <RealMathBadge label="Real Client Math & Workers" />
            <SimulatedBadge label="Mock Engine Backend Ready" />
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white leading-tight">
            Image Steganography Research & Demonstration Platform
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed">
            An empirical workbench designed to analyze cover image compatibility, simulate high-entropy payload embedding, extract confidential messages, and evaluate carrier distortion through real mathematical metrics.
          </p>

          {/* Hero Action CTAs */}
          <div className="flex flex-wrap items-center gap-4 pt-2">
            <button
              type="button"
              onClick={handleStartDemo}
              className="inline-flex items-center gap-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-500 hover:to-indigo-500 text-white font-bold text-sm shadow-xl shadow-brand-500/25 border border-brand-400/30 transition-all active:scale-95"
            >
              <Sparkles className="w-4 h-4" />
              <span>Launch Complete Interactive Demo</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              type="button"
              onClick={() => navigate('/compatibility')}
              className="inline-flex items-center gap-2 px-5 py-3.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-sm border border-slate-700 transition-all"
            >
              <ShieldCheck className="w-4 h-4 text-brand-400" />
              <span>Analyze Custom Cover</span>
            </button>
          </div>
        </div>
      </div>

      {/* Core Concepts: Steganography & Dual Modes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="p-8 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-4">
          <div className="w-10 h-10 rounded-xl bg-brand-500/20 text-brand-400 flex items-center justify-center">
            <Lock className="w-5 h-5" />
          </div>
          <h2 className="text-xl font-bold text-white">What is Image Steganography?</h2>
          <p className="text-sm text-slate-300 leading-relaxed">
            While cryptography encrypts data to make it unintelligible, <strong>steganography conceals the very existence of the communication</strong> by embedding secret data inside innocuous carrier media (such as photographic pixel buffers).
          </p>
          <p className="text-xs text-slate-400 leading-relaxed">
            High-fidelity steganography balances three competing constraints: <strong>Embedding Capacity</strong> (how much data fits), <strong>Fidelity/Imperceptibility</strong> (maintaining high PSNR/SSIM), and <strong>Steganalysis Security</strong> (resisting statistical detection by SRM classifiers).
          </p>
        </div>

        <div className="p-8 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-4">
          <div className="w-10 h-10 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center">
            <Binary className="w-5 h-5" />
          </div>
          <h2 className="text-xl font-bold text-white">Exact vs. Approximate Mode</h2>
          <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-850">
              <strong className="text-brand-300 block mb-0.5">Exact-Image / Discrete LSB Mode:</strong>
              Bit-for-bit mathematical equality. Every pixel of the secret payload is recovered losslessly with zero tolerance for degradation.
            </div>
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-850">
              <strong className="text-teal-300 block mb-0.5">Approximate Neural Autoencoder Mode:</strong>
              Deep learning autoencoders compress images into high-dimensional latent spaces and reconstruct them approximately with perceptual decoders.
            </div>
          </div>
        </div>
      </div>

      {/* Six-Stage Steganographic Workflow */}
      <div className="flex flex-col gap-6">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">The GuardianPixel Research Workflow</h2>
          <p className="text-sm text-slate-400 mt-1">
            Follow the empirical research cycle from cover compatibility validation to extraction and residual differential analysis.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {workflowSteps.map((wf) => (
            <button
              key={wf.step}
              type="button"
              onClick={() => navigate(wf.path)}
              className="text-left p-6 rounded-2xl bg-slate-900/60 hover:bg-slate-900 border border-slate-800 hover:border-brand-500/40 transition-all group flex flex-col justify-between min-h-[170px]"
            >
              <div>
                <span className="font-mono text-xs font-bold text-brand-400 tracking-wider block mb-2">
                  PHASE {wf.step}
                </span>
                <h3 className="text-base font-semibold text-white group-hover:text-brand-300 transition-colors">
                  {wf.title}
                </h3>
                <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
                  {wf.desc}
                </p>
              </div>

              <div className="flex items-center gap-1 text-xs font-medium text-brand-400 mt-4 group-hover:translate-x-1 transition-transform">
                <span>Explore Workflow</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Feature Showcase Grid */}
      <div className="p-8 rounded-3xl bg-slate-900/40 border border-slate-800/80 flex flex-col gap-6">
        <h2 className="text-xl font-bold text-white">Workbench Modules</h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div
            onClick={() => navigate('/compatibility')}
            className="p-5 rounded-xl bg-slate-950 border border-slate-850 hover:border-slate-700 cursor-pointer transition-all"
          >
            <ShieldCheck className="w-6 h-6 text-brand-400 mb-3" />
            <h3 className="text-sm font-semibold text-white">Compatibility Analysis</h3>
            <p className="text-xs text-slate-400 mt-1">
              Texture score, capacity math & headroom ratios.
            </p>
          </div>

          <div
            onClick={() => navigate('/embed')}
            className="p-5 rounded-xl bg-slate-950 border border-slate-850 hover:border-slate-700 cursor-pointer transition-all"
          >
            <Cpu className="w-6 h-6 text-indigo-400 mb-3" />
            <h3 className="text-sm font-semibold text-white">Embedding Engine</h3>
            <p className="text-xs text-slate-400 mt-1">
              Text, file, and exact secret image embedding modes.
            </p>
          </div>

          <div
            onClick={() => navigate('/compare')}
            className="p-5 rounded-xl bg-slate-950 border border-slate-850 hover:border-slate-700 cursor-pointer transition-all"
          >
            <Sliders className="w-6 h-6 text-teal-400 mb-3" />
            <h3 className="text-sm font-semibold text-white">Comparative Audit</h3>
            <p className="text-xs text-slate-400 mt-1">
              Split slider, real PSNR/SSIM, & amplified error diffs.
            </p>
          </div>

          <div
            onClick={() => navigate('/research')}
            className="p-5 rounded-xl bg-slate-950 border border-slate-850 hover:border-slate-700 cursor-pointer transition-all"
          >
            <BarChart3 className="w-6 h-6 text-amber-400 mb-3" />
            <h3 className="text-sm font-semibold text-white">Research Dashboard</h3>
            <p className="text-xs text-slate-400 mt-1">
              6 Recharts benchmarks across 120+ seeded trials.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
