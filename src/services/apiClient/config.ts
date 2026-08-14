/**
 * API Client Configuration
 * 
 * When a real backend is available, flip USE_MOCK_API to false
 * and update API_BASE_URL. No frontend components or service wrappers
 * will require any changes.
 */

export const USE_MOCK_API = true;

export const API_BASE_URL = 'https://api.stegolab.internal/v1';

export const SIMULATED_LATENCY_MIN_MS = 250;
export const SIMULATED_LATENCY_MAX_MS = 950;
