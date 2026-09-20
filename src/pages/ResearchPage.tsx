import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  BarChart3,
  Database,
  RefreshCw,
} from 'lucide-react';
import { apiClient } from '../services/apiClient/client';

interface MetricSummary {
  mean: number;
  median: number;
  min: number;
  max: number;
}

interface HistogramSummary {
  red: MetricSummary;
  green: MetricSummary;
  blue: MetricSummary;
  mean: MetricSummary;
}

interface SteganalysisMetrics {
  lsb_chi_square_statistic_change: MetricSummary;
  lsb_chi_square_p_value_change: MetricSummary;
  rs_combined_imbalance_change: MetricSummary;
  modified_pixel_count: MetricSummary;
  changed_pixel_ratio: MetricSummary;
  smooth_pixel_fraction: MetricSummary;
  smooth_region_change_ratio: MetricSummary;
  histogram_l1: HistogramSummary;
}

interface ValidationInfo {
  dataset: string;
  image_count: number;
  image_ids: string[];
  payload_bytes: number;
  embedding_output: string;
  analysis_type: string;
  interpretation_note: string;
}

interface PerImageResult {
  image_id?: string;
  id?: string;
  [key: string]: any;
}

interface SteganalysisResponse {
  isSimulated: boolean;
  validation: ValidationInfo;
  metrics: SteganalysisMetrics;
  perImage: PerImageResult[];

  // Backend may return this as either a string or an array.
  limitations?: string | string[];
}

const formatNumber = (value: number, digits = 6) => {
  if (!Number.isFinite(value)) {
    return '—';
  }

  return value.toFixed(digits);
};

const formatPercent = (value: number, digits = 2) => {
  if (!Number.isFinite(value)) {
    return '—';
  }

  return `${(value * 100).toFixed(digits)}%`;
};

const formatBytes = (bytes: number) => {
  if (!Number.isFinite(bytes)) {
    return '—';
  }

  if (bytes < 1024) {
    return `${bytes} B`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(2)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};

const getLimitations = (
  limitations?: string | string[]
): string[] => {
  if (!limitations) {
    return [];
  }

  if (Array.isArray(limitations)) {
    return limitations;
  }

  return [limitations];
};

export const ResearchPage: React.FC = () => {
  const [result, setResult] =
    useState<SteganalysisResponse | null>(null);

  const [loading, setLoading] = useState(true);

  const [error, setError] =
    useState<string | null>(null);

  const loadResults = async () => {
    setLoading(true);
    setError(null);

    const response =
      await apiClient.get<SteganalysisResponse>(
        '/steganalysis'
      );

    if (response.error) {
      setError(response.error.message);
      setResult(null);
    } else {
      setResult(response.data ?? null);
    }

    setLoading(false);
  };

  useEffect(() => {
    loadResults();
  }, []);

  const perImage = result?.perImage ?? [];

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 text-center">
          <RefreshCw className="w-6 h-6 text-brand-400 animate-spin mx-auto mb-3" />

          <p className="text-sm text-slate-300">
            Loading steganalysis validation results...
          </p>
        </div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="p-6 rounded-2xl bg-red-950/30 border border-red-900/50">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400" />

            <div>
              <h2 className="text-sm font-semibold text-red-300">
                Unable to load steganalysis results
              </h2>

              <p className="text-xs text-red-400/80 mt-1">
                {error ??
                  'No result data was returned by the backend.'}
              </p>
            </div>
          </div>

          <button
            onClick={loadResults}
            className="mt-4 inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-700"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  const { validation, metrics } = result;

  const limitations = getLimitations(
    result.limitations
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-8">

      {/* Research status */}
      <div className="p-4 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-start gap-3">
        <BarChart3 className="w-5 h-5 text-emerald-400 mt-0.5 flex-shrink-0" />

        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-sm text-emerald-300">
              REAL VALIDATION RESULTS
            </span>

            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-300">
              {validation.analysis_type}
            </span>
          </div>

          <p className="text-xs text-emerald-400/80 mt-1">
            Results are loaded from the GuardianPixel backend
            validation dataset rather than a synthetic research
            dataset.
          </p>
        </div>
      </div>

      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <BarChart3 className="w-6 h-6 text-brand-400" />

            <h1 className="text-2xl font-bold tracking-tight text-white">
              Steganalysis Validation
            </h1>
          </div>

          <p className="text-sm text-slate-400 max-w-3xl">
            Classical steganalysis measurements across the
            GuardianPixel validation set.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-slate-400 bg-slate-900 px-3 py-2 rounded-xl border border-slate-800">
          <Database className="w-4 h-4 text-brand-400" />

          <span>
            Images:{' '}
            <strong className="text-white">
              {validation.image_count}
            </strong>
          </span>
        </div>
      </div>

      {/* Validation configuration */}
      <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">

        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">
            Dataset
          </span>

          <p className="text-sm font-semibold text-white mt-2">
            {validation.dataset}
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">
            Validation Images
          </span>

          <p className="text-xl font-bold font-mono text-brand-300 mt-2">
            {validation.image_count}
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">
            Payload
          </span>

          <p className="text-xl font-bold font-mono text-teal-400 mt-2">
            {formatBytes(validation.payload_bytes)}
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">
            Output
          </span>

          <p className="text-xl font-bold font-mono text-emerald-400 mt-2 uppercase">
            {validation.embedding_output}
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800">
          <span className="text-[11px] font-semibold text-slate-400 uppercase">
            Changed Pixels
          </span>

          <p className="text-xl font-bold font-mono text-amber-400 mt-2">
            {formatPercent(
              metrics.changed_pixel_ratio.mean
            )}
          </p>

          <p className="text-[10px] text-slate-500 mt-1">
            mean ratio
          </p>
        </div>

      </section>

      {/* Main metrics */}
      <section>

        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-semibold text-white">
              Steganalysis Indicators
            </h2>

            <p className="text-xs text-slate-500 mt-1">
              Aggregate statistics across the{' '}
              {validation.image_count}-image validation set.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">

          <MetricCard
            label="Modified Pixels"
            value={metrics.modified_pixel_count.mean.toLocaleString(
              undefined,
              {
                maximumFractionDigits: 0,
              }
            )}
            detail={`Median: ${metrics.modified_pixel_count.median.toLocaleString(
              undefined,
              {
                maximumFractionDigits: 0,
              }
            )}`}
          />

          <MetricCard
            label="Changed Pixel Ratio"
            value={formatPercent(
              metrics.changed_pixel_ratio.mean
            )}
            detail={`Range: ${formatPercent(
              metrics.changed_pixel_ratio.min
            )} – ${formatPercent(
              metrics.changed_pixel_ratio.max
            )}`}
            accent="text-emerald-400"
          />

          <MetricCard
            label="LSB χ² Change"
            value={formatNumber(
              metrics.lsb_chi_square_statistic_change.mean,
              2
            )}
            detail={`Median: ${formatNumber(
              metrics.lsb_chi_square_statistic_change.median,
              2
            )}`}
            accent="text-brand-300"
          />

          <MetricCard
            label="RS Imbalance Change"
            value={metrics.rs_combined_imbalance_change.mean.toExponential(
              2
            )}
            detail={`Median: ${metrics.rs_combined_imbalance_change.median.toExponential(
              2
            )}`}
            accent="text-purple-300"
          />

          <MetricCard
            label="Smooth Region Change"
            value={formatPercent(
              metrics.smooth_region_change_ratio.mean
            )}
            detail={`Median: ${formatPercent(
              metrics.smooth_region_change_ratio.median
            )}`}
            accent="text-amber-400"
          />

          <MetricCard
            label="Smooth Pixel Fraction"
            value={formatPercent(
              metrics.smooth_pixel_fraction.mean
            )}
            detail="Quantile-based region"
            accent="text-teal-400"
          />

          <MetricCard
            label="Histogram L1"
            value={formatNumber(
              metrics.histogram_l1.mean.mean,
              7
            )}
            detail={`Median: ${formatNumber(
              metrics.histogram_l1.mean.median,
              7
            )}`}
            accent="text-indigo-300"
          />

          <MetricCard
            label="LSB p-value Change"
            value={metrics.lsb_chi_square_p_value_change.mean.toExponential(
              2
            )}
            detail="Interpret with sample size"
            accent="text-slate-300"
          />

        </div>
      </section>

      {/* Histogram channel metrics */}
      <section className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800">

        <div className="mb-5">
          <h2 className="text-base font-semibold text-white">
            Histogram Distribution Change
          </h2>

          <p className="text-xs text-slate-400 mt-1">
            Mean L1 distance between cover and stego channel
            histograms.
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">

          <ChannelMetric
            channel="Red"
            value={metrics.histogram_l1.red.mean}
          />

          <ChannelMetric
            channel="Green"
            value={metrics.histogram_l1.green.mean}
          />

          <ChannelMetric
            channel="Blue"
            value={metrics.histogram_l1.blue.mean}
          />

          <ChannelMetric
            channel="RGB Mean"
            value={metrics.histogram_l1.mean.mean}
          />

        </div>
      </section>

      {/* Per image table */}
      <section className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800">

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">

          <div>
            <h2 className="text-base font-semibold text-white">
              Per-Image Validation Results
            </h2>

            <p className="text-xs text-slate-400 mt-1">
              Individual measurements for each validation image.
            </p>
          </div>

          <span className="text-xs font-mono text-slate-500">
            {perImage.length} records
          </span>

        </div>

        <div className="overflow-x-auto max-h-[500px]">

          <table className="w-full text-left text-xs">

            <thead className="sticky top-0 bg-slate-950 border-b border-slate-800 text-slate-400 font-mono">

              <tr>
                <th className="p-2.5">Image</th>
                <th className="p-2.5">Modified Pixels</th>
                <th className="p-2.5">Changed Ratio</th>
                <th className="p-2.5">LSB χ² Change</th>
                <th className="p-2.5">RS Change</th>
                <th className="p-2.5">Smooth Change</th>
              </tr>

            </thead>

            <tbody className="divide-y divide-slate-800/60 font-mono">

              {perImage.map((row, index) => (

                <tr
                  key={
                    row.image_id ??
                    row.id ??
                    index
                  }
                  className="hover:bg-slate-800/30"
                >

                  <td className="p-2.5 text-brand-300 font-semibold">
                    {row.image_id ??
                      row.id ??
                      `Image ${index + 1}`}
                  </td>

                  <td className="p-2.5 text-slate-300">
                    {Number.isFinite(
                      Number(row.modified_pixel_count)
                    )
                      ? Number(
                          row.modified_pixel_count
                        ).toLocaleString()
                      : '—'}
                  </td>

                  <td className="p-2.5 text-emerald-400">
                    {Number.isFinite(
                      Number(row.changed_pixel_ratio)
                    )
                      ? formatPercent(
                          Number(
                            row.changed_pixel_ratio
                          )
                        )
                      : '—'}
                  </td>

                  <td className="p-2.5 text-slate-300">
                    {Number.isFinite(
                      Number(
                        row.lsb_chi_square_statistic_change
                      )
                    )
                      ? Number(
                          row.lsb_chi_square_statistic_change
                        ).toFixed(2)
                      : '—'}
                  </td>

                  <td className="p-2.5 text-purple-300">
                    {Number.isFinite(
                      Number(
                        row.rs_combined_imbalance_change
                      )
                    )
                      ? Number(
                          row.rs_combined_imbalance_change
                        ).toExponential(2)
                      : '—'}
                  </td>

                  <td className="p-2.5 text-amber-300">
                    {Number.isFinite(
                      Number(
                        row.smooth_region_change_ratio
                      )
                    )
                      ? formatPercent(
                          Number(
                            row.smooth_region_change_ratio
                          )
                        )
                      : '—'}
                  </td>

                </tr>

              ))}

            </tbody>

          </table>

        </div>
      </section>

      {/* Interpretation */}
      <section className="p-6 rounded-2xl bg-amber-500/5 border border-amber-500/20">

        <div className="flex items-start gap-3">

          <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5 flex-shrink-0" />

          <div>

            <h2 className="text-sm font-semibold text-amber-300">
              Interpretation & Limitations
            </h2>

            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              {validation.interpretation_note}
            </p>

            {limitations.map((limitation, index) => (
              <p
                key={index}
                className="text-xs text-slate-500 mt-2 leading-relaxed"
              >
                • {limitation}
              </p>
            ))}

          </div>

        </div>

      </section>

      {/* Validation summary */}
      <section className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800">

        <h2 className="text-base font-semibold text-white mb-4">
          Validation Summary
        </h2>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">

          <SummaryItem
            label="Changed Ratio Min"
            value={formatPercent(
              metrics.changed_pixel_ratio.min
            )}
          />

          <SummaryItem
            label="Changed Ratio Median"
            value={formatPercent(
              metrics.changed_pixel_ratio.median
            )}
          />

          <SummaryItem
            label="Changed Ratio Max"
            value={formatPercent(
              metrics.changed_pixel_ratio.max
            )}
          />

          <SummaryItem
            label="Images Analysed"
            value={String(validation.image_count)}
          />

        </div>

      </section>

      {/* Refresh */}
      <div className="flex justify-end">

        <button
          onClick={loadResults}
          className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-xs text-slate-300 border border-slate-800"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh Results
        </button>

      </div>

    </div>
  );
};

interface MetricCardProps {
  label: string;
  value: string;
  detail: string;
  accent?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  detail,
  accent = 'text-white',
}) => (
  <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between">

    <span className="text-[11px] font-semibold text-slate-400 uppercase">
      {label}
    </span>

    <span
      className={`text-xl font-bold font-mono my-2 ${accent}`}
    >
      {value}
    </span>

    <span className="text-[10px] text-slate-500 font-mono">
      {detail}
    </span>

  </div>
);

const ChannelMetric: React.FC<{
  channel: string;
  value: number;
}> = ({ channel, value }) => (
  <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">

    <span className="text-xs font-semibold text-slate-400">
      {channel}
    </span>

    <p className="text-lg font-bold font-mono text-brand-300 mt-2">
      {formatNumber(value, 7)}
    </p>

  </div>
);

const SummaryItem: React.FC<{
  label: string;
  value: string;
}> = ({ label, value }) => (
  <div>

    <span className="text-slate-500">
      {label}
    </span>

    <p className="text-slate-200 font-mono font-semibold mt-1">
      {value}
    </p>

  </div>
);