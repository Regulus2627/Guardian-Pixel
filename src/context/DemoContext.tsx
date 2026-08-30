import React, { createContext, useContext, useState, useCallback } from 'react';
import { CoverImageMeta, StegoResult } from '../types';
import { getUnifiedDemoData } from '../utils/sampleData';
import { extractDataUrlMetadata } from '../utils/imageProcessing';

interface DemoContextType {
  coverDataUrl: string | null;
  coverMeta: CoverImageMeta | null;
  coverTitle: string;
  secretImageDataUrl: string | null;
  payloadText: string;
  passphrase: string;
  stegoResult: StegoResult | null;
  payloadKind: 'text' | 'file' | 'image';
  uploadedFile: { name: string; size: number; type: string; dataUrl: string } | null;
  setCover: (dataUrl: string, meta: CoverImageMeta, title?: string) => void;
  setSecretImage: (dataUrl: string) => void;
  setPayloadText: (text: string) => void;
  setPassphrase: (pass: string) => void;
  setStegoResult: (result: StegoResult | null) => void;
  setPayloadKind: (kind: 'text' | 'file' | 'image') => void;
  setUploadedFile: (file: { name: string; size: number; type: string; dataUrl: string } | null) => void;
  loadDemo: () => Promise<void>;
  resetAll: () => void;
}

const DemoContext = createContext<DemoContextType | null>(null);

export const DemoProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [coverDataUrl, setCoverDataUrl] = useState<string | null>(null);
  const [coverMeta, setCoverMeta] = useState<CoverImageMeta | null>(null);
  const [coverTitle, setCoverTitle] = useState<string>('Custom Cover');
  const [secretImageDataUrl, setSecretImageDataUrl] = useState<string | null>(null);
  const [payloadText, setPayloadTextState] = useState<string>('');
  const [passphrase, setPassphraseState] = useState<string>('');
  const [stegoResult, setStegoResult] = useState<StegoResult | null>(null);
  const [payloadKind, setPayloadKind] = useState<'text' | 'file' | 'image'>('text');
  const [uploadedFile, setUploadedFile] = useState<{ name: string; size: number; type: string; dataUrl: string } | null>(null);

  const setCover = useCallback((dataUrl: string, meta: CoverImageMeta, title = 'Cover Image') => {
    setCoverDataUrl(dataUrl);
    setCoverMeta(meta);
    setCoverTitle(title);
  }, []);

  const setSecretImage = useCallback((dataUrl: string) => {
    setSecretImageDataUrl(dataUrl);
  }, []);

  const setPayloadText = useCallback((text: string) => {
    setPayloadTextState(text);
  }, []);

  const setPassphrase = useCallback((pass: string) => {
    setPassphraseState(pass);
  }, []);

  const resetAll = useCallback(() => {
    setCoverDataUrl(null);
    setCoverMeta(null);
    setCoverTitle('Custom Cover');
    setSecretImageDataUrl(null);
    setPayloadTextState('');
    setPassphraseState('');
    setStegoResult(null);
    setPayloadKind('text');
    setUploadedFile(null);
  }, []);

  const loadDemo = useCallback(async () => {
    const demo = getUnifiedDemoData();
    const meta = await extractDataUrlMetadata(demo.coverImage.dataUrl, demo.coverImage.title);

    setCoverDataUrl(demo.coverImage.dataUrl);
    setCoverMeta(meta);
    setCoverTitle(demo.coverImage.title);
    setSecretImageDataUrl(demo.secretImage);
    setPayloadTextState(demo.sampleText);
    setPassphraseState(demo.samplePassphrase);
    setPayloadKind('text');
  }, []);

  return (
    <DemoContext.Provider
      value={{
        coverDataUrl,
        coverMeta,
        coverTitle,
        secretImageDataUrl,
        payloadText,
        passphrase,
        stegoResult,
        payloadKind,
        uploadedFile,
        setCover,
        setSecretImage,
        setPayloadText,
        setPassphrase,
        setStegoResult,
        setPayloadKind,
        setUploadedFile,
        loadDemo,
        resetAll,
      }}
    >
      {children}
    </DemoContext.Provider>
  );
};

export function useDemo() {
  const context = useContext(DemoContext);
  if (!context) {
    throw new Error('useDemo must be used within a DemoProvider');
  }
  return context;
}
