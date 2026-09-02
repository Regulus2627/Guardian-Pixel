import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Loader2,
  RefreshCw,
  XCircle,
} from 'lucide-react';

import { apiClient } from '../services/apiClient/client';

interface MethodSummary {
  method: string;
  total_runs: number;
  embedding_success_runs: number;
  reliable_ber_zero_runs: number;
  reliable_success_rate_percent: number;
  mean_ber: number | string;
  mean_psnr_all_embedded: number | string;
  mean_ssim_all_embedded: number | string;
  mean_psnr_reliable: number | string;
  mean_ssim_reliable: number | string;
  mean_changed_pixel_ratio: number | string;
  mean_embedding_seconds: number | string;
  mean_extraction_seconds: number | string;
}

interface CategorySummary {
  cover_category: string;
  method: string;
  total_runs: number;
  reliable_runs: number;
  reliable_success_rate_percent: number;
  mean_psnr_reliable: number | string;
  mean_ssim_reliable: number | string;
  mean_changed_pixel_ratio: number | string;
}

interface CaseMethod {
  method: string;
  extraction_success: boolean;
  ber: number | string;
  psnr: number | string;
  ssim: number | string;
  changed_pixels: number | string;
  embedding_seconds: number | string;
}

interface CompareResponse {
  isSimulated: boolean;
  preliminary: boolean;
  aggregate: {
    independentCovers: number;
    payloadLevels: number[];
    methodCount: number;
    methodResultRows: number;
    methods: string[];
    methodSummary: MethodSummary[];
    categorySummary: CategorySummary[];
    charts: Record<string, string>;
  };
  singleCaseStudy: null | {
    cover: string;
    width: number;
    height: number;
    payloadOriginalBytes: number;
    equalEmbeddedBits: number;
    equalTotalBpp: number;
    methods: CaseMethod[];
    differenceImages: Record<string, string>;
    importantNote: string;
  };
  conclusion: string;
  limitations: string;
}

function numeric(value: number | string): number | null {
  if (value === '' || value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function show(value: number | string, digits: number): string {
  const parsed = numeric(value);
  return parsed === null ? '—' : parsed.toFixed(digits);
}

function summaryRowClass(method: string, reliability: number): string {
  if (method === 'GuardianPixel') {
    return 'bg-emerald-500/10';
  }
  if (reliability === 0) {
    return 'bg-rose-500/10';
  }
  return 'bg-slate-950/50';
}

export const ComparePage: React.FC = () => {
  const [data, setData] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [category, setCategory] = useState<string>('All');

  const loadData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiClient.get<CompareResponse>('/compare');
      if (response.error) throw new Error(response.error.message);
      if (!response.data) throw new Error('No comparison result returned.');
      if (response.data.isSimulated) {
        throw new Error('The backend returned simulated comparison data.');
      }
      setData(response.data);
    } catch (caught: unknown) {
      setError(
        caught instanceof Error
          ? caught.message
          : 'Could not load comparison results.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const categories = useMemo(() => {
    if (!data) return [];
    return Array.from(
      new Set(data.aggregate.categorySummary.map((row) => row.cover_category))
    ).sort();
  }, [data]);

  const visibleCategoryRows = useMemo(() => {
    if (!data) return [];
    if (category === 'All') return data.aggregate.categorySummary;
    return data.aggregate.categorySummary.filter(
      (row) => row.cover_category === category
    );
  }, [data, category]);

  if (loading) {
    return (
      <div className="min-h-[500px] flex items-center justify-center text-slate-300">
        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
        Loading real multi-cover results...
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-8">
      <div className="border-b border-slate-800 pb-6 flex flex-col md:flex-row md:items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-brand-400" />
            <h1 className="text-2xl font-bold text-white">
              Multi-Cover Equal-BPP Baseline Evaluation
            </h1>
            <span className="text-[10px] px-2 py-1 rounded border border-amber-500/30 bg-amber-500/10 text-amber-300">
              Real Preliminary Evaluation
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-2 max-w-4xl">
            Aggregate results from multiple independent covers and payload levels,
            followed by one detailed visual case study.
          </p>
        </div>

        <button
          type="button"
          onClick={loadData}
          className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs border border-slate-700"
        >
          <RefreshCw className="w-4 h-4" /> Reload Results
        </button>
      </div>

      {error && (
        <div className="p-5 rounded-xl border border-rose-500/40 bg-rose-500/10 text-rose-200">
          <div className="flex items-center gap-2 font-semibold">
            <XCircle className="w-5 h-5" /> Real comparison result unavailable
          </div>
          <p className="text-sm mt-2">{error}</p>
        </div>
      )}

      {data && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <p className="text-[10px] uppercase text-slate-500">Independent covers</p>
              <p className="text-2xl font-bold text-white">
                {data.aggregate.independentCovers}
              </p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <p className="text-[10px] uppercase text-slate-500">Payload levels</p>
              <p className="text-2xl font-bold text-white">
                {data.aggregate.payloadLevels.length}
              </p>
              <p className="text-[10px] text-slate-500 mt-1">
                {data.aggregate.payloadLevels.map((value) => value.toLocaleString()).join(', ')} B
              </p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <p className="text-[10px] uppercase text-slate-500">Methods</p>
              <p className="text-2xl font-bold text-white">
                {data.aggregate.methodCount}
              </p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <p className="text-[10px] uppercase text-slate-500">Method-result rows</p>
              <p className="text-2xl font-bold text-white">
                {data.aggregate.methodResultRows}
              </p>
            </div>
          </div>

          <section>
            <h2 className="text-lg font-bold text-white mb-3">
              Aggregate Method Summary
            </h2>
            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-xs">
                <thead className="bg-slate-900 text-slate-300">
                  <tr>
                    <th className="text-left p-3">Method</th>
                    <th className="p-3">Reliable Runs</th>
                    <th className="p-3">Success Rate</th>
                    <th className="p-3">Mean BER</th>
                    <th className="p-3">Reliable PSNR</th>
                    <th className="p-3">Reliable SSIM</th>
                    <th className="p-3">Changed Pixels</th>
                    <th className="p-3">Embed Time</th>
                    <th className="p-3">Extract Time</th>
                  </tr>
                </thead>
                <tbody>
                  {data.aggregate.methodSummary.map((row) => (
                    <tr
                      key={row.method}
                      className={`border-t border-slate-800 ${summaryRowClass(
                        row.method,
                        row.reliable_success_rate_percent
                      )}`}
                    >
                      <td className="p-3 font-semibold text-white">{row.method}</td>
                      <td className="p-3 text-center font-mono text-slate-200">
                        {row.reliable_ber_zero_runs}/{row.total_runs}
                      </td>
                      <td className="p-3 text-center font-mono text-slate-200">
                        {row.reliable_success_rate_percent.toFixed(1)}%
                      </td>
                      <td className="p-3 text-center font-mono text-slate-200">
                        {show(row.mean_ber, 6)}
                      </td>
                      <td className="p-3 text-center font-mono text-slate-200">
                        {show(row.mean_psnr_reliable, 4)} dB
                      </td>
                      <td className="p-3 text-center font-mono text-slate-200">
                        {show(row.mean_ssim_reliable, 6)}
                      </td>
                      <td className="p-3 text-center font-mono text-slate-200">
                        {show(row.mean_changed_pixel_ratio, 4)}%
                      </td>
                      <td className="p-3 text-center font-mono text-slate-200">
                        {show(row.mean_embedding_seconds, 3)} s
                      </td>
                      <td className="p-3 text-center font-mono text-slate-200">
                        {show(row.mean_extraction_seconds, 3)} s
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section>
            <h2 className="text-lg font-bold text-white mb-3">
              Real Multi-Cover Charts
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {(['psnr', 'ssim', 'reliability'] as const).map((chartName) => (
                <div
                  key={chartName}
                  className="p-3 rounded-xl bg-white border border-slate-800"
                >
                  {data.aggregate.charts[chartName] ? (
                    <img
                      src={data.aggregate.charts[chartName]}
                      alt={`${chartName} comparison chart`}
                      className="w-full h-auto object-contain"
                    />
                  ) : (
                    <div className="h-52 flex items-center justify-center text-slate-500">
                      Chart unavailable
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>

          <section>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
              <div>
                <h2 className="text-lg font-bold text-white">
                  Results by Cover Category
                </h2>
                <p className="text-xs text-slate-400">
                  Compare Smooth, Mixed and Textured cover groups.
                </p>
              </div>

              <select
                value={category}
                onChange={(event) => setCategory(event.target.value)}
                className="bg-slate-900 border border-slate-700 text-slate-200 rounded-lg px-3 py-2 text-xs"
              >
                <option value="All">All categories</option>
                {categories.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </div>

            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-xs">
                <thead className="bg-slate-900 text-slate-300">
                  <tr>
                    <th className="text-left p-3">Category</th>
                    <th className="text-left p-3">Method</th>
                    <th className="p-3">Reliable Runs</th>
                    <th className="p-3">Success Rate</th>
                    <th className="p-3">Mean PSNR</th>
                    <th className="p-3">Mean SSIM</th>
                    <th className="p-3">Changed Pixels</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleCategoryRows.map((row) => (
                    <tr
                      key={`${row.cover_category}-${row.method}`}
                      className="border-t border-slate-800 bg-slate-950/40"
                    >
                      <td className="p-3 font-semibold text-brand-300">
                        {row.cover_category}
                      </td>
                      <td className="p-3 text-white">{row.method}</td>
                      <td className="p-3 text-center font-mono text-slate-200">
                        {row.reliable_runs}/{row.total_runs}
                      </td>
                      <td className="p-3 text-center font-mono text-slate-200">
                        {row.reliable_success_rate_percent.toFixed(1)}%
                      </td>
                      <td className="p-3 text-center font-mono text-slate-200">
                        {show(row.mean_psnr_reliable, 4)} dB
                      </td>
                      <td className="p-3 text-center font-mono text-slate-200">
                        {show(row.mean_ssim_reliable, 6)}
                      </td>
                      <td className="p-3 text-center font-mono text-slate-200">
                        {show(row.mean_changed_pixel_ratio, 4)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <div className="p-5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-100">
            <p className="font-semibold">Aggregate interpretation</p>
            <p className="text-xs mt-1 leading-relaxed">{data.conclusion}</p>
          </div>

          <div className="p-5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-100 flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 flex-shrink-0" />
            <div>
              <p className="font-semibold">Evaluation limitation</p>
              <p className="text-xs mt-1 leading-relaxed">{data.limitations}</p>
            </div>
          </div>

          {data.singleCaseStudy && (
            <section className="flex flex-col gap-4 border-t border-slate-800 pt-7">
              <div>
                <h2 className="text-lg font-bold text-white">
                  Single-Cover Spatial Case Study
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  {data.singleCaseStudy.cover} — {data.singleCaseStudy.width} ×{' '}
                  {data.singleCaseStudy.height} — same{' '}
                  {data.singleCaseStudy.equalEmbeddedBits.toLocaleString()} bits — BPP{' '}
                  {data.singleCaseStudy.equalTotalBpp.toFixed(6)}
                </p>
              </div>

              <p className="text-xs text-slate-400">
                Bright/green pixels are ×255 amplified differences. Actual
                maximum channel change is only ±1.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
                {data.singleCaseStudy.methods.map((method) => {
                  const image = data.singleCaseStudy?.differenceImages[method.method];
                  return (
                    <div
                      key={method.method}
                      className={`rounded-xl border p-3 ${
                        method.method === 'GuardianPixel'
                          ? 'border-emerald-500/40 bg-emerald-500/10'
                          : !method.extraction_success
                          ? 'border-rose-500/40 bg-rose-500/10'
                          : 'border-slate-800 bg-slate-900'
                      }`}
                    >
                      <p className="text-xs font-semibold text-white text-center min-h-[32px]">
                        {method.method}
                      </p>
                      <div className="h-40 mt-2 bg-black rounded-lg overflow-hidden flex items-center justify-center">
                        {image ? (
                          <img
                            src={image}
                            alt={`${method.method} difference map`}
                            className="w-full h-full object-contain"
                          />
                        ) : (
                          <span className="text-[10px] text-slate-600">No image</span>
                        )}
                      </div>
                      <p className="text-[10px] text-center text-slate-400 mt-2">
                        BER {show(method.ber, 3)}
                      </p>
                    </div>
                  );
                })}
              </div>

              <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-xs text-rose-200">
                <strong>Canny synchronization finding:</strong>{' '}
                {data.singleCaseStudy.importantNote}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
};
