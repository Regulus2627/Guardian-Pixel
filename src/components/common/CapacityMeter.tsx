import React from 'react';
import { CapacityResult } from '../../types';
import { formatBytes } from '../../utils/imageProcessing';
import { RealMathBadge } from './RealMathBadge';
import { AlertTriangle, CheckCircle2, AlertOctagon } from 'lucide-react';

interface CapacityMeterProps {
  capacity: CapacityResult;
  showDetails?: boolean;
}

export const CapacityMeter: React.FC<CapacityMeterProps> = ({ capacity, showDetails = true }) => {
  const { maxTheoreticalBytes, recommendedSafeBytes, currentPayloadBytes, utilizationPct } = capacity;

  // Visual status
  let statusColor = 'bg-emerald-500';
  let statusText = 'Optimal Headroom';
  let statusBadge = 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30';
  let StatusIcon = CheckCircle2;

  if (utilizationPct > 100) {
    statusColor = 'bg-rose-500';
    statusText = 'Capacity Exceeded';
    statusBadge = 'bg-rose-500/10 text-rose-300 border-rose-500/30';
    StatusIcon = AlertOctagon;
  } else if (utilizationPct > 75) {
    statusColor = 'bg-amber-500';
    statusText = 'Approaching Safe Limit';
    statusBadge = 'bg-amber-500/10 text-amber-300 border-amber-500/30';
    StatusIcon = AlertTriangle;
  }

  // Safe percentage bar representation capped at 100% for the visual width
  const visualPct = Math.min(100, Math.max(0, utilizationPct));

  return (
    <div className="w-full bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-sm">
      <div className="flex items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Carrier Capacity Utilization
          </span>
          <RealMathBadge label="Real Math" />
        </div>

        <div className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border ${statusBadge}`}>
          <StatusIcon className="w-3.5 h-3.5" />
          <span>{statusText} ({utilizationPct}%)</span>
        </div>
      </div>

      {/* Progress Track */}
      <div className="relative w-full h-3 bg-slate-800 rounded-full overflow-hidden mb-2">
        <div
          className={`h-full transition-all duration-300 rounded-full ${statusColor}`}
          style={{ width: `${visualPct}%` }}
        />
        {/* Recommended safe threshold indicator mark at 100% of safe limit */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white/40 shadow"
          style={{ left: '100%', transform: 'translateX(-1px)' }}
          title="100% Recommended Safe Limit"
        />
      </div>

      {showDetails && (
        <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-slate-800/80 text-xs">
          <div>
            <span className="text-slate-400 block text-[11px]">Current Payload</span>
            <span className="font-mono font-semibold text-slate-200 tabular-nums">
              {formatBytes(currentPayloadBytes)}
            </span>
          </div>

          <div>
            <span className="text-slate-400 block text-[11px]">Safe Recommendation</span>
            <span className="font-mono font-semibold text-brand-300 tabular-nums">
              {formatBytes(recommendedSafeBytes)}
            </span>
          </div>

          <div>
            <span className="text-slate-400 block text-[11px]">Theoretical Max LSB</span>
            <span className="font-mono font-semibold text-slate-400 tabular-nums">
              {formatBytes(maxTheoreticalBytes)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
