# StegoLab Development Progress

## Overview
A comprehensive research and demonstration frontend workbench for image steganography, cover compatibility analysis, payload embedding/extraction, and scientific evaluation.

---

## Progress Tracker

| Milestone | Status | Details |
|---|---|---|
| **1. Scaffolding & Setup** | ✅ Complete | React 18, TypeScript, Tailwind CSS, Vite, Recharts, Lucide, routing skeleton |
| **2. Mock API Client Architecture** | ✅ Complete | `services/apiClient/` with `USE_MOCK_API = true`, mock handlers for `/embed`, `/extract`, `/crypto/estimate`, `/compression/estimate`, `/metrics/detect` |
| **3. Real Metadata, Capacity & Sample Data** | ✅ Complete | `utils/imageProcessing.ts`, `utils/capacity.ts`, `utils/sampleData.ts` with unified "Load Demo" action |
| **4. Web Worker Texture/Edge Suite** | ✅ Complete | Sobel edge approximation, local Shannon entropy, spatial variance, fusion mask, Otsu binary segmentation |
| **5. Compatibility Page (`/compatibility`)** | ✅ Complete | Real cover validation, TextEncoder UTF-8 meter, real headroom/texture classification & expandable "Why?" math |
| **6. Embed Page (`/embed`)** | ✅ Complete | Text mode, File mode, Exact Image mode + real error diff map, ephemeral passphrase security in `finally` |
| **7. Extract Page (`/extract`)** | ✅ Complete | Stego upload, passphrase auth, simulated extraction, safe error states (wrong password, corrupt, etc.) |
| **8. Compare Page (`/compare`)** | ✅ Complete | Real PSNR, SSIM, MSE, split slider, amplified difference map ($1\times - 50\times$), channel breakdown table |
| **9. Research Dashboard (`/research`)** | ✅ Complete | Persistent Demo banner, 6 Recharts visualizations, deterministic 120+ benchmark dataset, CI/SD statistics |
| **10. Methodology Page (`/methodology`)** | ✅ Complete | Pipeline diagrams, Exact vs Autoencoder table, formatted math definitions, threat models, mock disclosures |
| **11. Build Verification & Polishing** | ✅ Complete | `npx tsc --noEmit` passed clean with 0 errors; `npm run build` bundled successfully |

---

## Mocked vs. Real Breakdown

### Real (Client-Side Computations):
- Canvas image metadata extraction (dimensions, color channels, format, byte size)
- UTF-8 payload byte counting via `TextEncoder`
- Theoretical and recommended safe capacity arithmetic & headroom ratios
- PSNR ($10 \log_{10}(255^2/\text{MSE})$), SSIM ($8\times 8$ windowed), and MSE calculations
- Per-pixel difference maps and amplified error visualizations with dynamic sliders
- Web Worker texture analysis (Shannon entropy, local variance, Sobel edge approximation, fusion, Otsu binary segmentation)
- Statistical calculations (Mean, Standard Deviation, 95% Confidence Intervals)

### Simulated (Behind Mock API Client with "Simulated" Badge):
- Deep neural embedding & extraction pipelines (`POST /embed`, `POST /extract`)
- Cryptographic key derivation and ciphertext size estimation (`POST /crypto/estimate`)
- Compression ratio estimation (`POST /compression/estimate`)
- Steganalysis detection risk score (`POST /metrics/detect`)
- Benchmark dataset (`isDemoData: true`)
