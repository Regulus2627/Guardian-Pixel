import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  ShieldCheck,
  Cpu,
  Layers,
  FileKey,
  Sliders,
  BarChart3,
  BookOpen,
  Sparkles,
  RotateCcw,
  Binary,
} from 'lucide-react';
import { useDemo } from '../../context/DemoContext';

interface NavItem {
  name: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
}

const navItems: NavItem[] = [
  { name: 'Overview', path: '/', icon: Layers },
  { name: 'Compatibility', path: '/compatibility', icon: ShieldCheck },
  { name: 'Embed', path: '/embed', icon: Cpu },
  { name: 'Extract', path: '/extract', icon: FileKey },
  { name: 'Compare', path: '/compare', icon: Sliders },
  { name: 'Research', path: '/research', icon: BarChart3, badge: 'Demo' },
  { name: 'Methodology', path: '/methodology', icon: BookOpen },
];

export const Navbar: React.FC = () => {
  const location = useLocation();
  const { loadDemo, resetAll } = useDemo();

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800 bg-slate-950/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-2">
          {/* Brand Logo */}
          <NavLink
            to="/"
            className="flex items-center gap-2.5 focus-visible:ring-2 focus-visible:ring-brand-400 rounded-lg p-1"
          >
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-brand-500 to-indigo-600 flex items-center justify-center text-white shadow-lg shadow-brand-500/20">
              <Binary className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-base tracking-tight text-white">GuardianPixel</span>
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-brand-500/20 text-brand-400 border border-brand-500/30">
                  v1.0
                </span>
              </div>
              <p className="text-[10px] text-slate-400 hidden sm:block">AI-Guided Secure Steganography</p>
            </div>
          </NavLink>

          {/* Desktop Navigation Links */}
          <nav className="hidden lg:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;

              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={`relative flex items-center gap-2 px-3 py-2 text-xs font-medium rounded-lg transition-all ${
                    isActive
                      ? 'text-white bg-slate-800/90 border border-slate-700 shadow-sm font-semibold'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                  }`}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-brand-400' : 'text-slate-500'}`} />
                  <span>{item.name}</span>
                  {item.badge && (
                    <span className="text-[9px] px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300 font-mono">
                      {item.badge}
                    </span>
                  )}
                  {isActive && (
                    <span className="absolute bottom-0 left-2 right-2 h-0.5 bg-brand-500 rounded-full" />
                  )}
                </NavLink>
              );
            })}
          </nav>

          {/* Action CTAs: Load Demo & Reset */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => loadDemo()}
              type="button"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-500 hover:to-indigo-500 text-white shadow-md shadow-brand-500/15 border border-brand-400/30 transition-all active:scale-95"
              title="Populate sample cover, secret image, text payload & parameters in one click"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Load Demo</span>
            </button>

            <button
              onClick={() => resetAll()}
              type="button"
              className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800 transition-all"
              title="Reset all loaded state and securely clear passwords"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Reset</span>
            </button>
          </div>
        </div>

        {/* Mobile Navigation Row */}
        <div className="lg:hidden flex items-center gap-1 overflow-x-auto py-2 border-t border-slate-900 scrollbar-none">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;

            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={`flex-shrink-0 flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-md transition-all ${
                  isActive
                    ? 'text-white bg-slate-800 border border-slate-700 font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-brand-400' : 'text-slate-500'}`} />
                <span>{item.name}</span>
              </NavLink>
            );
          })}
        </div>
      </div>
    </header>
  );
};
