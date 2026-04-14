/**
 * CausalStress API Client
 * Replaces mockData.ts with real API calls to the FastAPI backend.
 * 
 * Usage in components:
 *   import { api } from '@/services/api';
 *   const { data } = useQuery({ queryKey: ['dashboard'], queryFn: api.dashboard.getSummary });
 */

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

function buildUrl(path: string): string {
  if (!API_BASE) return path;
  return `${API_BASE}${path}`;
}

async function fetchJSON<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(buildUrl(path), {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function checkApiHealth(): Promise<{ status: string; service?: string; environment?: string }> {
  return fetchJSON('/live');
}

// ==================== TYPES ====================

export interface DashboardSummary {
  currentRegime: {
    name: string;
    confidence: number;
    streak: number;
    date: string | null;
    probabilities: Record<string, number>;
  };
  spx: { value: number | null; date: string | null };
  system: {
    variables: number;
    tradingDays: number;
    causalEdges: number;
    scenarios: number;
  };
}

export interface CausalNode {
  id: string;
  label: string;
  category: string;
  major?: boolean;
}

export interface CausalEdge {
  source: string;
  target: string;
  weight: number;
  regime?: string;
}

export interface CausalGraphResponse {
  nodes: CausalNode[];
  edges: CausalEdge[];
  stats: { totalNodes: number; totalEdges: number; method: string };
}

export interface RegimeSegment {
  regime: string;
  start: string;
  end: string;
  months: number;
}

export interface RegimeCharacteristic {
  regime: string;
  days: number;
  pct: number;
  vixMean: number;
  spxReturn: number;
  hySpread: number;
  yieldCurve: number;
}

export interface ScenarioImpact {
  variable: string;
  p5: number;
  median: number;
  p95: number;
  mean: number;
}

export interface ScenarioFanPoint {
  day: number;
  median: number;
  p5: number;
  p95: number;
  paths: number[];
}

export interface ScenarioFocusVariable {
  id: string;
  label: string;
  ticker: string;
}

export interface ScenarioSeriesPoint {
  day: number;
  median: number;
  p5: number;
  p95: number;
}

export interface ScenarioDistributionBucket {
  bucket: number;
  freq: number;
}

export interface ScenarioVariableResult {
  label: string;
  ticker: string;
  valueType: 'return' | 'level';
  distribution: ScenarioDistributionBucket[];
  fanChart: ScenarioSeriesPoint[];
}

export interface ScenarioStressRangeRow {
  variable: string;
  ticker: string;
  current: number | null;
  currentDate: string | null;
  valueType: 'return' | 'level';
  p5Move: number | null;
  medianMove: number | null;
  p95Move: number | null;
  meanMove: number | null;
  impliedMedian: number | null;
  impliedLow: number | null;
  impliedHigh: number | null;
}

export interface ScenarioResponse {
  id: string;
  model: string;
  modelSignature: string;
  family: {
    id: string;
    label: string;
    eventType: string;
  };
  severity: string;
  graph: string;
  filter: string;
  candidateCount: number;
  scenarioCount: number;
  horizon: number;
  createdAt?: string | null;
  avgPlausibility: number;
  plausibility: {
    mean: number;
    weightedMean: number;
    rawMean: number;
    min: number;
    max: number;
  };
  focusVariables: ScenarioFocusVariable[];
  shockTemplate: { label: string; ticker: string; shock: number }[];
  variables: Record<string, ScenarioVariableResult>;
  keyVariableStressRange: ScenarioStressRangeRow[];
}

export interface ScenarioGenerateRequest {
  family_id: string;
  severity: string;
  horizon: number;
  displayed_paths: number;
  anchor_variable_override?: string;
  anchor_magnitude_override?: number;
  random_seed?: number;
}

export interface StressTestResult {
  id: string;
  portfolioValue: number;
  scenario: {
    id: string;
    familyId: string;
    family: string;
    eventType: string;
    severity: string;
    model: string;
    generatedAt: string | null;
    pathsUsed: number;
    horizon: number;
  };
  var95: number;
  var99: number;
  cvar95: number;
  maxDrawdown: number;
  lossProbabilities: { any: number; over5: number; over10: number };
  sectorRisk: { sector: string; value: number; color: string }[];
  holdingRisk: { holding: string; risk: number }[];
  pnlDistribution: { bucket: string; freq: number }[];
  topContributors: { asset: string; contribution: number; pct: number }[];
  topAbsorbers: { asset: string; contribution: number; pct: number }[];
}

export interface ScenarioListItem {
  id: string;
  familyId: string;
  family: string;
  eventType: string;
  shockMagnitude: number;
  nScenarios: number;
  meanPlausibility: number;
  createdAt: string;
  severity: string;
  horizon: number;
}

export interface ChatResponse {
  role: string;
  content: string;
}

// ==================== API CLIENT ====================

export const api = {
  dashboard: {
    getSummary: (): Promise<DashboardSummary> =>
      fetchJSON('/api/v1/dashboard/summary'),

    getSpxHistory: (days = 180): Promise<{ date: string; value: number }[]> =>
      fetchJSON(`/api/v1/dashboard/spx-history?days=${days}`),

    getRegimeChart: (months = 27): Promise<any[]> =>
      fetchJSON(`/api/v1/dashboard/regime-chart?months=${months}`),

    getTopCausalLinks: (limit = 10): Promise<{ cause: string; effect: string; weight: number; confidence: number }[]> =>
      fetchJSON(`/api/v1/dashboard/top-causal-links?limit=${limit}`),
  },

  causal: {
    getGraph: (regime?: string, minWeight = 0): Promise<CausalGraphResponse> => {
      const params = new URLSearchParams();
      if (regime) params.set('regime', regime);
      if (minWeight > 0) params.set('min_weight', String(minWeight));
      return fetchJSON(`/api/v1/causal/graph?${params}`);
    },

    getRegimeComparison: (): Promise<Record<string, { edges: number }>> =>
      fetchJSON('/api/v1/causal/regime-comparison'),
  },

  regimes: {
    getCurrent: (): Promise<{ name: string; confidence: number; streak: number; date: string }> =>
      fetchJSON('/api/v1/regimes/current'),

    getTimeline: (): Promise<RegimeSegment[]> =>
      fetchJSON('/api/v1/regimes/timeline'),

    getCharacteristics: (): Promise<RegimeCharacteristic[]> =>
      fetchJSON('/api/v1/regimes/characteristics'),

    getTransitionMatrix: (): Promise<{ labels: string[]; data: number[][] }> =>
      fetchJSON('/api/v1/regimes/transition-matrix'),
  },

  scenarios: {
    getLatest: (eventType?: string): Promise<ScenarioResponse> => {
      const params = eventType ? `?event_type=${eventType}` : '';
      return fetchJSON(`/api/v1/scenarios/latest${params}`);
    },

    generate: (payload: ScenarioGenerateRequest): Promise<ScenarioResponse> =>
      fetchJSON('/api/v1/scenarios/generate', {
        method: 'POST',
        body: JSON.stringify(payload),
      }),

    list: (): Promise<ScenarioListItem[]> =>
      fetchJSON('/api/v1/scenarios/list'),
  },

  stressTest: {
    run: (holdings: any[], scenarioId?: string): Promise<StressTestResult> =>
      fetchJSON('/api/v1/stress-test/run', {
        method: 'POST',
        body: JSON.stringify({
          holdings,
          scenario_id: scenarioId || null,
        }),
      }),
  },

  advisor: {
    chat: (message: string, history: { role: string; content: string }[] = []): Promise<ChatResponse> =>
      fetchJSON('/api/v1/advisor/chat', {
        method: 'POST',
        body: JSON.stringify({ message, history }),
      }),

    getSuggestedPrompts: (): Promise<string[]> =>
      fetchJSON('/api/v1/advisor/suggested-prompts'),
  },
};

// ==================== FALLBACK TO MOCK DATA ====================
// If the API is unreachable, components can fall back to mockData
// This allows the frontend to work both with and without the backend

export async function fetchWithFallback<T>(
  apiFn: () => Promise<T>,
  fallback: T,
): Promise<T> {
  try {
    return await apiFn();
  } catch {
    console.warn('API unavailable, using mock data');
    return fallback;
  }
}
