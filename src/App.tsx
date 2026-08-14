import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { DemoProvider } from './context/DemoContext';
import { Navbar } from './components/common/Navbar';
import { Footer } from './components/common/Footer';
import { HomePage } from './pages/HomePage';
import { CompatibilityPage } from './pages/CompatibilityPage';
import { EmbedPage } from './pages/EmbedPage';
import { ExtractPage } from './pages/ExtractPage';
import { ComparePage } from './pages/ComparePage';
import { ResearchPage } from './pages/ResearchPage';
import { MethodologyPage } from './pages/MethodologyPage';

export const App: React.FC = () => {
  return (
    <DemoProvider>
      <div className="flex flex-col min-h-screen bg-slate-950 text-slate-100">
        <Navbar />
        <main className="flex-1 w-full">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/compatibility" element={<CompatibilityPage />} />
            <Route path="/embed" element={<EmbedPage />} />
            <Route path="/extract" element={<ExtractPage />} />
            <Route path="/compare" element={<ComparePage />} />
            <Route path="/research" element={<ResearchPage />} />
            <Route path="/methodology" element={<MethodologyPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </DemoProvider>
  );
};
