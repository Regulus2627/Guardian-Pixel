import { USE_MOCK_API, API_BASE_URL, SIMULATED_LATENCY_MIN_MS, SIMULATED_LATENCY_MAX_MS } from './config';
import {
  mockEmbedHandler,
  mockExtractHandler,
  mockCryptoEstimateHandler,
  mockCompressionEstimateHandler,
  mockDetectionScoreHandler,
} from './mockHandlers';

export interface ApiResponse<T> {
  data?: T;
  error?: {
    code: string;
    message: string;
    status?: number;
  };
}

/**
 * Simulates network latency scaled to request payload size
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Central API Client interface
 * Swaps seamlessly between Mock Handlers and Real HTTP REST Endpoints
 */
export const apiClient = {
  async post<T = any>(endpoint: string, body: any): Promise<ApiResponse<T>> {
    if (!USE_MOCK_API) {
      // Real backend fetch branch
      try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          return {
            error: {
              code: errData.code || `HTTP_${response.status}`,
              message: errData.message || response.statusText,
              status: response.status,
            },
          };
        }
        const data = await response.json();
        return { data };
      } catch (err: any) {
        return {
          error: {
            code: 'NETWORK_ERROR',
            message: err.message || 'Failed to communicate with remote steganography server',
          },
        };
      }
    }

    // Mock API Dispatcher branch
    const delay = Math.floor(
      SIMULATED_LATENCY_MIN_MS + Math.random() * (SIMULATED_LATENCY_MAX_MS - SIMULATED_LATENCY_MIN_MS)
    );
    await sleep(delay);

    try {
      let result: any;
      switch (endpoint) {
        case '/embed':
        case '/stego/embed':
          result = await mockEmbedHandler(body);
          break;

        case '/extract':
        case '/stego/extract':
          result = await mockExtractHandler(body);
          break;

        case '/crypto/estimate':
          result = await mockCryptoEstimateHandler(body);
          break;

        case '/compression/estimate':
          result = await mockCompressionEstimateHandler(body);
          break;

        case '/metrics/detect':
          result = await mockDetectionScoreHandler(body);
          break;

        default:
          return {
            error: {
              code: 'NOT_FOUND',
              message: `Mock endpoint '${endpoint}' is not defined.`,
            },
          };
      }
      return { data: result as T };
    } catch (err: any) {
      return {
        error: {
          code: err.code || 'INTERNAL_ERROR',
          message: err.message || 'An error occurred during mock service execution.',
        },
      };
    }
  },

  async get<T = any>(endpoint: string): Promise<ApiResponse<T>> {
    if (!USE_MOCK_API) {
      try {
        const res = await fetch(`${API_BASE_URL}${endpoint}`);
        if (!res.ok) {
          return { error: { code: `HTTP_${res.status}`, message: res.statusText } };
        }
        const data = await res.json();
        return { data };
      } catch (err: any) {
        return { error: { code: 'NETWORK_ERROR', message: err.message } };
      }
    }

    await sleep(200);
    return {
      error: { code: 'NOT_IMPLEMENTED', message: `Mock GET for ${endpoint} not required.` },
    };
  },
};
