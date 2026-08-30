import React from 'react';
import { RealMathBadge } from './RealMathBadge';
import { SimulatedBadge } from './SimulatedBadge';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  isSimulated?: boolean;
  subtext?: string;
  icon?: React.ComponentType<{ className?: string }>;
  tone?: 'default' | 'emerald' | 'amber' | 'blue' | 'purple';
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  isSimulated = false,
  subtext,
  icon: Icon,
  tone = 'default',
  className = '',
}) => {
  let borderClass = 'border-slate-800 bg-slate-900/80';
  let valueColor = 'text-white';

  if (tone === 'emerald') {
    borderClass = 'border-emerald-500/30 bg-emerald-950/20';
    valueColor = 'text-emerald-300';
  } else if (tone === 'amber') {
    borderClass = 'border-amber-500/30 bg-amber-950/20';
    valueColor = 'text-amber-300';
  } else if (tone === 'blue') {
    borderClass = 'border-brand-500/30 bg-brand-950/20';
    valueColor = 'text-brand-300';
  } else if (tone === 'purple') {
    borderClass = 'border-indigo-500/30 bg-indigo-950/20';
    valueColor = 'text-indigo-300';
  }

  return (
    <div className={`p-4 rounded-xl border ${borderClass} relative flex flex-col justify-between ${className}`}>
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          {Icon && <Icon className="w-4 h-4 text-slate-400" />}
          <span className="text-xs font-semibold text-slate-400 tracking-wide uppercase">{label}</span>
        </div>

        {isSimulated ? (
          <SimulatedBadge />
        ) : (
          <RealMathBadge />
        )}
      </div>

      <div className="flex items-baseline gap-1 my-1">
        <span className={`text-2xl font-bold font-mono tracking-tight tabular-nums ${valueColor}`}>
          {value}
        </span>
        {unit && <span className="text-xs text-slate-400 font-mono">{unit}</span>}
      </div>

      {subtext && <p className="text-[11px] text-slate-400 mt-1 leading-snug">{subtext}</p>}
    </div>
  );
};
