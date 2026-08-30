import React from 'react';
import { ShieldCheck, Database, FileCode2 } from 'lucide-react';
import { USE_MOCK_API } from '../../services/apiClient/config';

export const Footer: React.FC = () => {
  return (
    <footer className="w-full border-t border-slate-850 bg-slate-950/80 text-slate-400 py-8 px-4 sm:px-6 lg:px-8 mt-auto">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-xs">
        <div className="flex flex-col sm:flex-row items-center gap-3">
          <div className="flex items-center gap-1.5 text-slate-300 font-medium">
            <span className="w-2 h-2 rounded-full bg-brand-500 animate-pulse"></span>
            StegoLab Scientific Evaluation Platform
          </div>
          <span className="hidden sm:inline text-slate-700">|</span>
          <p className="text-slate-500 text-center sm:text-left">
            Empirical Cover Compatibility & Spatial Steganalysis Workbench
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-4 text-slate-400 font-mono text-[11px]">
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-slate-900 border border-slate-800">
            <Database className="w-3 h-3 text-brand-400" />
            <span>Transport: {USE_MOCK_API ? 'Simulated (Mock API Client)' : 'Live REST Backend'}</span>
          </div>

          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-slate-900 border border-slate-800">
            <ShieldCheck className="w-3 h-3 text-emerald-400" />
            <span>Passphrase Storage: Ephemeral Only</span>
          </div>

          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-slate-900 border border-slate-800">
            <FileCode2 className="w-3 h-3 text-indigo-400" />
            <span>Workers: Active (Multi-threaded)</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
