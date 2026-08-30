import React from 'react';
import { Calculator } from 'lucide-react';

interface RealMathBadgeProps {
  label?: string;
  size?: 'sm' | 'md';
  className?: string;
}

export const RealMathBadge: React.FC<RealMathBadgeProps> = ({
  label = 'Client Math',
  size = 'sm',
  className = '',
}) => {
  const sizeClasses = size === 'sm' ? 'text-[10px] px-1.5 py-0.5' : 'text-xs px-2 py-0.5';

  return (
    <span
      className={`inline-flex items-center gap-1 font-medium tracking-wide rounded border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 ${sizeClasses} ${className}`}
      title="Computed directly in browser via client-side mathematical algorithms and pixel processing."
    >
      <Calculator className={size === 'sm' ? 'w-2.5 h-2.5' : 'w-3 h-3'} />
      {label}
    </span>
  );
};
