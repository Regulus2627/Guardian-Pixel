import React, { useState, useMemo } from 'react';
import {
  generateDeterministicDataset,
  computeDatasetStatistics,
  CATEGORIES,
  METHODS,
} from '../services/research/datasetService';
import { SimulatedBadge } from '../components/common/SimulatedBadge';
import { RealMathBadge } from '../components/common/RealMathBadge';
import { formatBytes } from '../utils/imageProcessing';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from 'recharts';
import {
  BarChart3,
  Database,
  AlertTriangle,
  Search,
  RotateCcw,
} from 'lucide-react';

export const ResearchPage: React.FC = () => {
  // Load full seeded deterministic dataset once
  const fullDataset = useMemo(() => generateDeterministicDataset(48291), []);

  // Filter States
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [selectedMethod, setSelectedMethod] = useState<string>('All');
  const [maxBpp, setMaxBpp] = useState<number>(1.0);
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Filtered rows
  const filteredRows = useMemo(() => {
    return fullDataset.filter((row) => {
      if (selectedCategory !== 'All' && row.category !== selectedCategory) return false;
      if (selectedMethod !== 'All' && row.method !== selectedMethod) return false;
      if (row.bpp > maxBpp) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        const matches =
          row.id?.toLowerCase().includes(q) ||
          row.method.toLowerCase().includes(q) ||
          row.category.toLowerCase().includes(q) ||
          row.resolution.toLowerCase().includes(q);
        if (!matches) return false;
      }
      return true;
    });
  }, [fullDataset, selectedCategory, selectedMethod, maxBpp, searchQuery]);

  // Real dynamic statistical calculations from filtered subset
  const stats = useMemo(() => computeDatasetStatistics(filteredRows), [filteredRows]);

  // Chart 1 & 2: PSNR & SSIM vs bpp grouped by method
  const pnsrSsimData = useMemo(() => {
    const bppBuckets = [0.05, 0.15, 0.35, 0.75];
    return bppBuckets.map((bpp) => {
      const point: any = { bpp: `${bpp} bpp` };
      METHODS.forEach((m) => {
        const matching = filteredRows.filter(
          (r) => r.method === m && Math.abs(r.bpp - bpp) < 0.1
        );
        if (matching.length > 0) {
          point[`psnr_${m}`] = Number(
            (matching.reduce((s, r) => s + r.psnr, 0) / matching.length).toFixed(1)
          );
          point[`ssim_${m}`] = Number(
            (matching.reduce((s, r) => s + r.ssim, 0) / matching.length).toFixed(4)
          );
          point[`detect_${m}`] = Number(
            ((matching.reduce((s, r) => s + r.detectionScore, 0) / matching.length) * 100).toFixed(1)
          );
        }
      });
      return point;
    });
  }, [filteredRows]);

  // Chart 4: Safe Capacity by Category
  const capacityByCategoryData = useMemo(() => {
    return CATEGORIES.map((cat) => {
      const rowsInCat = fullDataset.filter((r) => r.category === cat);
      const avgCap = rowsInCat.length > 0
        ? rowsInCat.reduce((s, r) => s + r.safeCapacityBytes, 0) / rowsInCat.length
        : 0;
      return {
        category: cat,
        safeCapKB: Number((avgCap / 1024).toFixed(1)),
      };
    });
  }, [fullDataset]);

  // Chart 6: Radar Comparison across Methods
  const radarData = useMemo(() => {
    return [
      { metric: 'PSNR Fidelity', 'LSB-Sequential': 85, 'Adaptive-Edge-LSB': 88, 'DCT-Frequency-Domain': 74, 'StegEx-ExactNet': 95, 'DenseAutoencoder-Approx': 65 },
      { metric: 'SSIM Structure', 'LSB-Sequential': 88, 'Adaptive-Edge-LSB': 90, 'DCT-Frequency-Domain': 78, 'StegEx-ExactNet': 97, 'DenseAutoencoder-Approx': 72 },
      { metric: 'Steganalysis Resistance', 'LSB-Sequential': 35, 'Adaptive-Edge-LSB': 78, 'DCT-Frequency-Domain': 65, 'StegEx-ExactNet': 92, 'DenseAutoencoder-Approx': 68 },
      { metric: 'Capacity Scaling', 'LSB-Sequential': 70, 'Adaptive-Edge-LSB': 65, 'DCT-Frequency-Domain': 50, 'StegEx-ExactNet': 85, 'DenseAutoencoder-Approx': 92 },
      { metric: 'Compute Speed', 'LSB-Sequential': 98, 'Adaptive-Edge-LSB': 82, 'DCT-Frequency-Domain': 65, 'StegEx-ExactNet': 55, 'DenseAutoencoder-Approx': 40 },
    ];
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-8">
      {/* Persistent Demo / Synthetic Data Banner */}
      <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-200 flex items-center justify-between gap-3 shadow-lg">
        <div className="flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0" />
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-sm text-amber-300">DEMO / SYNTHETIC RESEARCH DATASET</span>
              <SimulatedBadge label="Deterministic Pseudo-Random Seed" />
            </div>
            <p className="text-xs text-amber-400/90 mt-0.5">
              The benchmark figures and charts below represent a seeded deterministic simulation suite across 120+ benchmark conditions. They demonstrate analytical visualization workflows and do not reflect empirically measured neural network runs.
            </p>
          </div>
        </div>
      </div>

      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-850 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <BarChart3 className="w-6 h-6 text-brand-400" />
            <h1 className="text-2xl font-bold tracking-tight text-white">Steganographic Research Benchmark Lab</h1>
          </div>
          <p className="text-sm text-slate-400 max-w-3xl">
            Interactive comparative analysis of spatial, transform-domain, and neural steganography methods across multi-class cover imagery.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-slate-400 bg-slate-900 px-3 py-2 rounded-xl border border-slate-800 self-start md:self-auto">
          <Database className="w-4 h-4 text-brand-400" />
          <span>Active Rows: <strong className="text-white">{filteredRows.length}</strong> / {fullDataset.length}</span>
        </div>
      </div>

      {/* Interactive Filters Bar */}
      <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {/* Category filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400 font-medium">Category:</span>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 focus:border-brand-500"
            >
              <option value="All">All Categories (6)</option>
              {CATEGORIES.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* Method filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400 font-medium">Method:</span>
            <select
              value={selectedMethod}
              onChange={(e) => setSelectedMethod(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 focus:border-brand-500"
            >
              <option value="All">All Methods (5)</option>
              {METHODS.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          {/* bpp Slider */}
          <div className="flex items-center gap-2">
            <span className="text-slate-400 font-medium">Max bpp:</span>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={maxBpp}
              onChange={(e) => setMaxBpp(parseFloat(e.target.value))}
              className="w-24 h-1.5 bg-slate-800 rounded accent-brand-500 cursor-pointer"
            />
            <span className="font-mono text-brand-300 font-semibold">{maxBpp.toFixed(2)}</span>
          </div>
        </div>

        {/* Search & Reset */}
        <div className="flex items-center gap-2">
          <div className="relative flex-1 sm:w-60">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search experiments..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-600 focus:border-brand-500"
            />
          </div>

          <button
            type="button"
            onClick={() => {
              setSelectedCategory('All');
              setSelectedMethod('All');
              setMaxBpp(1.0);
              setSearchQuery('');
            }}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 transition-colors"
            title="Reset Filters"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Dynamic Statistical Metrics Ribbon */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">Mean PSNR (dB)</span>
          <span className="text-xl font-bold font-mono text-emerald-400 my-1">{stats.meanPsnr} dB</span>
          <span className="text-[10px] text-slate-500 font-mono">SD: ±{stats.sdPsnr} | 95% CI: [{stats.ci95Psnr[0]}, {stats.ci95Psnr[1]}]</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">Mean SSIM</span>
          <span className="text-xl font-bold font-mono text-brand-300 my-1">{stats.meanSsim}</span>
          <span className="text-[10px] text-slate-500 font-mono">SD: ±{stats.sdSsim} | 95% CI: [{stats.ci95Ssim[0]}, {stats.ci95Ssim[1]}]</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">Mean MSE</span>
          <span className="text-xl font-bold font-mono text-slate-200 my-1">{stats.meanMse}</span>
          <span className="text-[10px] text-slate-500 font-mono">SD: ±{stats.sdMse}</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">Mean Target bpp</span>
          <span className="text-xl font-bold font-mono text-teal-400 my-1">{stats.meanBpp} bpp</span>
          <span className="text-[10px] text-slate-500">Bits per carrier pixel</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">Mean Detection Risk</span>
          <span className="text-xl font-bold font-mono text-amber-400 my-1">{(stats.meanDetection * 100).toFixed(1)}%</span>
          <span className="text-[10px] text-slate-500">SRM Steganalysis prob.</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">Mean Exec Time</span>
          <span className="text-xl font-bold font-mono text-indigo-400 my-1">{stats.meanExecTime} ms</span>
          <span className="text-[10px] text-slate-500">Per embedding cycle</span>
        </div>
      </div>

      {/* 6 Recharts Interactive Visualization Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Chart 1: PSNR vs bpp */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white">1. PSNR Image Fidelity vs. Embedding Rate</h3>
              <p className="text-xs text-slate-400 mt-0.5">Higher is better. Values &gt; 40 dB indicate imperceptible distortion.</p>
            </div>
            <SimulatedBadge />
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={pnsrSsimData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="bpp" stroke="#64748b" fontSize={11} />
                <YAxis domain={[30, 60]} stroke="#64748b" fontSize={11} unit="dB" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="psnr_StegEx-ExactNet" name="StegEx-ExactNet" stroke="#38bdf8" strokeWidth={2.5} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="psnr_Adaptive-Edge-LSB" name="Adaptive-Edge-LSB" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="psnr_LSB-Sequential" name="LSB-Sequential" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="psnr_DCT-Frequency-Domain" name="DCT-Frequency" stroke="#818cf8" strokeWidth={1.5} strokeDasharray="4 4" />
                <Line type="monotone" dataKey="psnr_DenseAutoencoder-Approx" name="DenseAutoencoder" stroke="#f43f5e" strokeWidth={1.5} strokeDasharray="2 2" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: SSIM vs bpp */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white">2. Structural Similarity (SSIM) Degradation</h3>
              <p className="text-xs text-slate-400 mt-0.5">Measures perceptual preservation of luminance, contrast, and structure.</p>
            </div>
            <SimulatedBadge />
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={pnsrSsimData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="bpp" stroke="#64748b" fontSize={11} />
                <YAxis domain={[0.92, 1.0]} stroke="#64748b" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="ssim_StegEx-ExactNet" name="StegEx-ExactNet" stroke="#38bdf8" strokeWidth={2.5} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="ssim_Adaptive-Edge-LSB" name="Adaptive-Edge-LSB" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="ssim_LSB-Sequential" name="LSB-Sequential" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="ssim_DCT-Frequency-Domain" name="DCT-Frequency" stroke="#818cf8" strokeWidth={1.5} />
                <Line type="monotone" dataKey="ssim_DenseAutoencoder-Approx" name="DenseAutoencoder" stroke="#f43f5e" strokeWidth={1.5} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 3: Detection Score vs bpp */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white">3. Steganalysis Detection Probability (SRM)</h3>
              <p className="text-xs text-slate-400 mt-0.5">Lower is better. Risk threshold: &gt; 50% indicates detectable signature.</p>
            </div>
            <SimulatedBadge />
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={pnsrSsimData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="bpp" stroke="#64748b" fontSize={11} />
                <YAxis domain={[0, 100]} stroke="#64748b" fontSize={11} unit="%" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="detect_StegEx-ExactNet" name="StegEx-ExactNet" stroke="#38bdf8" strokeWidth={2.5} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="detect_Adaptive-Edge-LSB" name="Adaptive-Edge-LSB" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="detect_LSB-Sequential" name="LSB-Sequential" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="detect_DCT-Frequency-Domain" name="DCT-Frequency" stroke="#818cf8" strokeWidth={1.5} />
                <Line type="monotone" dataKey="detect_DenseAutoencoder-Approx" name="DenseAutoencoder" stroke="#f43f5e" strokeWidth={1.5} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 4: Safe Capacity by Category */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white">4. Safe Embedding Capacity by Carrier Class</h3>
              <p className="text-xs text-slate-400 mt-0.5">Average recommended safe payload capacity per category (in KB).</p>
            </div>
            <RealMathBadge label="Model Math" />
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={capacityByCategoryData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="category" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={11} unit=" KB" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Bar dataKey="safeCapKB" name="Safe Capacity (KB)" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 5: Execution Time vs Payload Size */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white">5. Execution Latency vs. Payload Size</h3>
              <p className="text-xs text-slate-400 mt-0.5">Embedding pipeline runtime complexity scaling.</p>
            </div>
            <SimulatedBadge />
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="payloadBytes"
                  name="Payload"
                  stroke="#64748b"
                  fontSize={10}
                  tickFormatter={(v) => `${Math.round(v / 1024)}KB`}
                />
                <YAxis dataKey="execTimeMs" name="Time" stroke="#64748b" fontSize={11} unit="ms" />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Scatter name="Benchmarks" data={filteredRows.slice(0, 50)} fill="#a855f7" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 6: Multi-Dimensional Radar Comparison */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white">6. Comprehensive Method Architecture Radar</h3>
              <p className="text-xs text-slate-400 mt-0.5">Normalized score matrix across fidelity, security, capacity, and speed.</p>
            </div>
            <SimulatedBadge />
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="metric" stroke="#94a3b8" fontSize={10} />
                <PolarRadiusAxis stroke="#475569" angle={30} domain={[0, 100]} />
                <Radar name="StegEx-ExactNet" dataKey="StegEx-ExactNet" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.4} />
                <Radar name="Adaptive-Edge-LSB" dataKey="Adaptive-Edge-LSB" stroke="#10b981" fill="#10b981" fillOpacity={0.25} />
                <Radar name="LSB-Sequential" dataKey="LSB-Sequential" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.2} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Tooltip contentStyle={{ backgroundColor: '#090d16', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Detailed Statistical Experiment Records Table */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-base font-semibold text-white">Filtered Benchmark Experiment Suite ({filteredRows.length} rows)</h3>
            <p className="text-xs text-slate-400">All metrics computed from deterministic simulated trial instances.</p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 font-mono">Confidence Level: 95% (Z=1.96)</span>
          </div>
        </div>

        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-slate-950 border-b border-slate-800 text-slate-400 font-mono">
              <tr>
                <th className="p-2.5">ID</th>
                <th className="p-2.5">Method</th>
                <th className="p-2.5">Category</th>
                <th className="p-2.5">Resolution</th>
                <th className="p-2.5">Payload</th>
                <th className="p-2.5">bpp</th>
                <th className="p-2.5">PSNR (dB)</th>
                <th className="p-2.5">SSIM</th>
                <th className="p-2.5">MSE</th>
                <th className="p-2.5">Detection Risk</th>
                <th className="p-2.5">Exec Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filteredRows.slice(0, 100).map((row) => (
                <tr key={row.id} className="hover:bg-slate-800/30">
                  <td className="p-2.5 text-slate-400 font-semibold">{row.id}</td>
                  <td className="p-2.5 text-brand-300 font-sans font-medium">{row.method}</td>
                  <td className="p-2.5 text-slate-300 font-sans">{row.category}</td>
                  <td className="p-2.5 text-slate-400">{row.resolution}</td>
                  <td className="p-2.5 text-slate-300">{formatBytes(row.payloadBytes)}</td>
                  <td className="p-2.5 text-emerald-400 font-semibold">{row.bpp}</td>
                  <td className="p-2.5 text-slate-200">{row.psnr}</td>
                  <td className="p-2.5 text-slate-200">{row.ssim}</td>
                  <td className="p-2.5 text-slate-400">{row.mse}</td>
                  <td className="p-2.5 text-amber-400">{(row.detectionScore * 100).toFixed(1)}%</td>
                  <td className="p-2.5 text-slate-400">{row.execTimeMs} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
