// ==================== TYPES ====================
export interface CausalNode {
  id: string;
  label: string;
  category: 'equity' | 'macro' | 'rates' | 'volatility' | 'commodities' | 'fixed-income' | 'currency';
  major?: boolean;
}

export interface CausalEdge {
  source: string;
  target: string;
  weight: number;
  regime?: string;
}

export interface RegimeSegment {
  regime: 'Calm' | 'Normal' | 'Elevated' | 'Stressed' | 'Crisis';
  start: string;
  end: string;
  months: number;
}

export interface ScenarioResult {
  day: number;
  median: number;
  p5: number;
  p95: number;
  paths: number[];
}

// ==================== CATEGORY COLORS ====================
export const categoryColors: Record<string, string> = {
  equity: '#3b82f6',
  macro: '#10b981',
  rates: '#f59e0b',
  volatility: '#ef4444',
  commodities: '#eab308',
  'fixed-income': '#a855f7',
  currency: '#06b6d4',
};

export const regimeColors: Record<string, string> = {
  Calm: '#10b981',
  Normal: '#22d3ee',
  Elevated: '#f59e0b',
  Stressed: '#ef4444',
  'High Stress': '#fb7185',
  Crisis: '#991b1b',
};

// ==================== CAUSAL GRAPH ====================
export const causalNodes: CausalNode[] = [
  { id: '^GSPC', label: 'S&P 500', category: 'equity', major: true },
  { id: '^NDX', label: 'Nasdaq 100', category: 'equity' },
  { id: '^RUT', label: 'Russell 2000', category: 'equity' },
  { id: 'XLK', label: 'Tech ETF', category: 'equity' },
  { id: 'XLF', label: 'Financial ETF', category: 'equity' },
  { id: 'XLE', label: 'Energy ETF', category: 'equity' },
  { id: 'XLV', label: 'Health ETF', category: 'equity' },
  { id: 'XLY', label: 'Consumer Disc', category: 'equity' },
  { id: 'XLRE', label: 'Real Estate ETF', category: 'equity' },
  { id: 'XLU', label: 'Utilities ETF', category: 'equity' },
  { id: 'EEM', label: 'Emerging Mkts', category: 'equity' },
  { id: 'A191RL1Q225SBEA', label: 'Real GDP', category: 'macro' },
  { id: 'CPIAUCSL', label: 'CPI', category: 'macro' },
  { id: 'PCEPILFE', label: 'Core PCE', category: 'macro' },
  { id: 'UNRATE', label: 'Unemployment', category: 'macro' },
  { id: 'PAYEMS', label: 'Nonfarm Payrolls', category: 'macro' },
  { id: 'INDPRO', label: 'Industrial Prod', category: 'macro' },
  { id: 'ICSA', label: 'Initial Claims', category: 'macro' },
  { id: 'M2SL', label: 'M2 Money Supply', category: 'macro' },
  { id: 'HOUST', label: 'Housing Starts', category: 'macro' },
  { id: 'UMCSENT', label: 'Consumer Sent', category: 'macro' },
  { id: 'RSXFS', label: 'Retail Sales', category: 'macro' },
  { id: 'DGS10', label: '10Y Treasury', category: 'rates', major: true },
  { id: 'DGS2', label: '2Y Treasury', category: 'rates' },
  { id: 'FEDFUNDS', label: 'Fed Funds Rate', category: 'rates' },
  { id: 'T10Y2Y', label: '10Y-2Y Spread', category: 'rates' },
  { id: '^VIX', label: 'VIX', category: 'volatility', major: true },
  { id: '^VVIX', label: 'VVIX', category: 'volatility' },
  { id: 'CL=F', label: 'Crude Oil', category: 'commodities' },
  { id: 'GC=F', label: 'Gold', category: 'commodities' },
  { id: 'TLT', label: '20Y Treasury Bond', category: 'fixed-income' },
  { id: 'LQD', label: 'IG Corp Bond', category: 'fixed-income' },
  { id: 'HYG', label: 'High Yield Bond', category: 'fixed-income' },
  { id: 'BAMLH0A0HYM2', label: 'HY OAS Spread', category: 'fixed-income' },
  { id: 'DX-Y.NYB', label: 'US Dollar Index', category: 'currency' },
  { id: 'EURUSD=X', label: 'EUR/USD', category: 'currency' },
  { id: 'JPYUSD=X', label: 'JPY/USD', category: 'currency' },
];

export const causalEdges: CausalEdge[] = [
  // ALL regime edges (always visible)
  { source: '^GSPC', target: 'XLF', weight: 2.0, regime: 'ALL' },
  { source: '^GSPC', target: 'XLV', weight: 1.83, regime: 'ALL' },
  { source: '^GSPC', target: 'XLU', weight: 1.72, regime: 'ALL' },
  { source: '^GSPC', target: 'XLE', weight: 1.61, regime: 'ALL' },
  { source: '^GSPC', target: '^NDX', weight: 0.93, regime: 'ALL' },
  { source: '^GSPC', target: '^RUT', weight: 0.90, regime: 'ALL' },
  { source: '^GSPC', target: 'XLY', weight: 0.69, regime: 'ALL' },
  { source: '^GSPC', target: 'EEM', weight: 0.81, regime: 'ALL' },
  { source: 'DGS10', target: 'T10Y2Y', weight: 1.42, regime: 'ALL' },
  { source: 'DGS2', target: 'T10Y2Y', weight: 1.29, regime: 'ALL' },
  { source: 'DGS10', target: 'DGS2', weight: 1.10, regime: 'ALL' },
  { source: 'DGS10', target: 'TLT', weight: 0.70, regime: 'ALL' },
  { source: 'PAYEMS', target: 'UNRATE', weight: 0.99, regime: 'ALL' },
  { source: 'PAYEMS', target: 'PCEPILFE', weight: 1.01, regime: 'ALL' },
  { source: '^VIX', target: '^VVIX', weight: 0.79, regime: 'ALL' },
  { source: '^NDX', target: 'XLK', weight: 0.70, regime: 'ALL' },
  { source: 'CL=F', target: 'CPIAUCSL', weight: 0.5, regime: 'ALL' },
  { source: 'FEDFUNDS', target: 'DGS2', weight: 0.45, regime: 'ALL' },
  { source: 'CPIAUCSL', target: 'PCEPILFE', weight: 0.63, regime: 'ALL' },
  { source: 'XLF', target: '^RUT', weight: 0.81, regime: 'ALL' },
  { source: 'BAMLH0A0HYM2', target: 'XLF', weight: 0.4, regime: 'ALL' },
  { source: 'DX-Y.NYB', target: 'EURUSD=X', weight: 0.69, regime: 'ALL' },
  { source: 'UNRATE', target: 'PAYEMS', weight: 0.82, regime: 'ALL' },
  { source: 'M2SL', target: 'CPIAUCSL', weight: 0.3, regime: 'ALL' },
  { source: 'HOUST', target: 'A191RL1Q225SBEA', weight: 0.2, regime: 'ALL' },
  { source: 'T10Y2Y', target: 'DGS2', weight: 0.77, regime: 'ALL' },
  { source: 'T10Y2Y', target: 'DGS10', weight: 0.70, regime: 'ALL' },
  { source: 'A191RL1Q225SBEA', target: 'A191RL1Q225SBEA', weight: 0.98, regime: 'ALL' },
  // Calm regime - fewer stress connections, stronger macro
  { source: '^GSPC', target: 'XLF', weight: 1.8, regime: 'Calm' },
  { source: '^GSPC', target: '^NDX', weight: 1.2, regime: 'Calm' },
  { source: 'DGS10', target: 'DGS2', weight: 1.5, regime: 'Calm' },
  { source: 'PAYEMS', target: 'UNRATE', weight: 0.7, regime: 'Calm' },
  { source: 'DGS10', target: 'TLT', weight: 0.9, regime: 'Calm' },
  { source: 'M2SL', target: 'CPIAUCSL', weight: 0.4, regime: 'Calm' },
  // Normal regime
  { source: '^GSPC', target: 'XLF', weight: 1.9, regime: 'Normal' },
  { source: '^GSPC', target: 'XLE', weight: 1.4, regime: 'Normal' },
  { source: '^GSPC', target: '^RUT', weight: 1.1, regime: 'Normal' },
  { source: 'DGS10', target: 'T10Y2Y', weight: 1.3, regime: 'Normal' },
  { source: 'PAYEMS', target: 'PCEPILFE', weight: 0.8, regime: 'Normal' },
  { source: 'CL=F', target: 'CPIAUCSL', weight: 0.6, regime: 'Normal' },
  { source: 'FEDFUNDS', target: 'DGS2', weight: 0.5, regime: 'Normal' },
  // Elevated regime - more cross-asset connections
  { source: '^VIX', target: '^GSPC', weight: 1.5, regime: 'Elevated' },
  { source: '^GSPC', target: 'XLF', weight: 2.1, regime: 'Elevated' },
  { source: '^GSPC', target: 'EEM', weight: 1.2, regime: 'Elevated' },
  { source: 'DGS10', target: 'T10Y2Y', weight: 1.6, regime: 'Elevated' },
  { source: 'BAMLH0A0HYM2', target: 'XLF', weight: 0.9, regime: 'Elevated' },
  { source: 'CL=F', target: 'CPIAUCSL', weight: 0.8, regime: 'Elevated' },
  { source: 'DX-Y.NYB', target: 'EURUSD=X', weight: 0.7, regime: 'Elevated' },
  // Stressed regime - high vol connections dominate
  { source: '^VIX', target: '^VVIX', weight: 1.8, regime: 'Stressed' },
  { source: '^VIX', target: '^GSPC', weight: 2.0, regime: 'Stressed' },
  { source: 'BAMLH0A0HYM2', target: 'XLF', weight: 1.4, regime: 'Stressed' },
  { source: '^GSPC', target: 'EEM', weight: 1.5, regime: 'Stressed' },
  { source: 'DGS10', target: 'TLT', weight: 1.3, regime: 'Stressed' },
  { source: 'FEDFUNDS', target: 'DGS2', weight: 1.1, regime: 'Stressed' },
  // Crisis regime - extreme correlations
  { source: '^VIX', target: '^GSPC', weight: 2.5, regime: 'Crisis' },
  { source: '^VIX', target: '^VVIX', weight: 2.2, regime: 'Crisis' },
  { source: 'BAMLH0A0HYM2', target: 'XLF', weight: 2.0, regime: 'Crisis' },
  { source: '^GSPC', target: 'EEM', weight: 2.0, regime: 'Crisis' },
  { source: '^GSPC', target: 'XLF', weight: 2.5, regime: 'Crisis' },
  { source: 'DGS10', target: 'TLT', weight: 1.8, regime: 'Crisis' },
];

export const topCausalLinks = [
  { cause: '^GSPC', effect: 'XLF', weight: 2.00, confidence: 100 },
  { cause: '^GSPC', effect: 'XLV', weight: 1.83, confidence: 100 },
  { cause: '^GSPC', effect: 'XLU', weight: 1.72, confidence: 100 },
  { cause: '^GSPC', effect: 'XLE', weight: 1.61, confidence: 100 },
  { cause: 'DGS10', effect: 'T10Y2Y', weight: 1.42, confidence: 100 },
  { cause: 'DGS2', effect: 'T10Y2Y', weight: 1.29, confidence: 100 },
  { cause: 'DGS10', effect: 'DGS2', weight: 1.10, confidence: 100 },
  { cause: 'PAYEMS', effect: 'UNRATE', weight: 0.99, confidence: 100 },
  { cause: '^GSPC', effect: '^NDX', weight: 0.93, confidence: 100 },
  { cause: '^GSPC', effect: '^RUT', weight: 0.90, confidence: 100 },
];

// ==================== REGIME DATA ====================
export const currentRegime = {
  name: 'Elevated' as const,
  confidence: 99.4,
  streak: 224,
  probabilities: {
    Calm: 0.0,
    Normal: 0.0,
    Elevated: 99.4,
    Stressed: 0.6,
    Crisis: 0.0,
  },
};

export const regimeTimeline: RegimeSegment[] = [
  { regime: 'Calm', start: '2005-01', end: '2007-06', months: 30 },
  { regime: 'Normal', start: '2007-07', end: '2007-12', months: 6 },
  { regime: 'Stressed', start: '2008-01', end: '2008-08', months: 8 },
  { regime: 'Crisis', start: '2008-09', end: '2009-06', months: 10 },
  { regime: 'Stressed', start: '2009-07', end: '2009-12', months: 6 },
  { regime: 'Normal', start: '2010-01', end: '2010-04', months: 4 },
  { regime: 'Stressed', start: '2010-05', end: '2010-07', months: 3 },
  { regime: 'Normal', start: '2010-08', end: '2011-06', months: 11 },
  { regime: 'Elevated', start: '2011-07', end: '2011-12', months: 6 },
  { regime: 'Normal', start: '2012-01', end: '2014-12', months: 36 },
  { regime: 'Elevated', start: '2015-01', end: '2015-09', months: 9 },
  { regime: 'Normal', start: '2015-10', end: '2017-12', months: 27 },
  { regime: 'Calm', start: '2018-01', end: '2018-01', months: 1 },
  { regime: 'Stressed', start: '2018-02', end: '2018-04', months: 3 },
  { regime: 'Normal', start: '2018-05', end: '2019-12', months: 20 },
  { regime: 'Calm', start: '2020-01', end: '2020-02', months: 2 },
  { regime: 'Crisis', start: '2020-03', end: '2020-04', months: 2 },
  { regime: 'Stressed', start: '2020-05', end: '2020-08', months: 4 },
  { regime: 'Normal', start: '2020-09', end: '2021-12', months: 16 },
  { regime: 'Elevated', start: '2022-01', end: '2022-10', months: 10 },
  { regime: 'Stressed', start: '2022-11', end: '2023-03', months: 5 },
  { regime: 'Normal', start: '2023-04', end: '2024-03', months: 12 },
  { regime: 'Calm', start: '2024-04', end: '2024-07', months: 4 },
  { regime: 'Normal', start: '2024-08', end: '2024-11', months: 4 },
  { regime: 'Elevated', start: '2024-12', end: '2026-03', months: 16 },
];

export const crisisAnnotations = [
  { year: 2008, label: 'GFC' },
  { year: 2010, label: 'Flash Crash' },
  { year: 2011, label: 'Euro Debt' },
  { year: 2015, label: 'China/Oil' },
  { year: 2018, label: 'Volmageddon' },
  { year: 2020, label: 'COVID' },
  { year: 2022, label: 'Rate Hikes' },
];

export const transitionMatrix = {
  labels: ['Calm', 'Normal', 'Elevated', 'Stressed', 'Crisis'],
  data: [
    [98.6, 0.0, 1.4, 0.0, 0.0],
    [0.1, 98.9, 0.0, 1.0, 0.0],
    [0.8, 0.0, 98.4, 0.8, 0.0],
    [0.0, 1.4, 1.0, 97.2, 0.4],
    [0.0, 0.0, 0.0, 1.3, 98.4],
  ],
};

export const regimeCharacteristics = [
  { regime: 'Calm', days: 1097, pct: 19.9, vixMean: 12.15, spxReturn: 0.079, hySpread: 3.52, yieldCurve: 0.30 },
  { regime: 'Normal', days: 1416, pct: 25.7, vixMean: 15.55, spxReturn: 0.060, hySpread: 5.14, yieldCurve: 1.87 },
  { regime: 'Elevated', days: 1591, pct: 28.9, vixMean: 17.82, spxReturn: 0.038, hySpread: 3.53, yieldCurve: 0.28 },
  { regime: 'Stressed', days: 1078, pct: 19.6, vixMean: 25.70, spxReturn: -0.023, hySpread: 6.33, yieldCurve: 1.27 },
  { regime: 'Crisis', days: 328, pct: 6.0, vixMean: 43.00, spxReturn: -0.116, hySpread: 12.53, yieldCurve: 1.70 },
];

// ==================== REGIME HISTORY CHART DATA ====================
// Monthly data from 2024-2026 for the dashboard regime chart
export const generateRegimeChartData = () => {
  const regimeMap: Record<string, number> = { Calm: 1, Normal: 2, Elevated: 3, Stressed: 4, Crisis: 5 };
  const segments = [
    { regime: 'Calm', startMonth: 0, endMonth: 3 },    // Jan-Apr 2024
    { regime: 'Normal', startMonth: 4, endMonth: 6 },   // May-Jul 2024
    { regime: 'Elevated', startMonth: 7, endMonth: 9 }, // Aug-Oct 2024
    { regime: 'Stressed', startMonth: 10, endMonth: 10 },// Nov 2024
    { regime: 'Normal', startMonth: 11, endMonth: 14 }, // Dec 2024-Mar 2025
    { regime: 'Calm', startMonth: 15, endMonth: 18 },   // Apr-Jul 2025
    { regime: 'Normal', startMonth: 19, endMonth: 21 }, // Aug-Oct 2025
    { regime: 'Elevated', startMonth: 22, endMonth: 26 },// Nov 2025-Mar 2026
  ];
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const data = [];
  for (let i = 0; i <= 26; i++) {
    const year = 2024 + Math.floor(i / 12);
    const monthIdx = i % 12;
    const seg = segments.find(s => i >= s.startMonth && i <= s.endMonth);
    const regime = seg?.regime || 'Normal';
    data.push({
      month: `${months[monthIdx]} ${year}`,
      monthShort: `${months[monthIdx].slice(0,3)} '${String(year).slice(2)}`,
      Calm: regime === 'Calm' ? regimeMap['Calm'] : 0,
      Normal: regime === 'Normal' ? regimeMap['Normal'] : 0,
      Elevated: regime === 'Elevated' ? regimeMap['Elevated'] : 0,
      Stressed: regime === 'Stressed' ? regimeMap['Stressed'] : 0,
      Crisis: regime === 'Crisis' ? regimeMap['Crisis'] : 0,
      regime,
      value: regimeMap[regime],
    });
  }
  return data;
};

// ==================== S&P CHART DATA ====================
export const generateSPData = () => {
  const data = [];
  let value = 5950;
  const start = new Date(2025, 9, 1);
  for (let i = 0; i < 180; i++) {
    const date = new Date(start);
    date.setDate(start.getDate() + i);
    value += (Math.random() - 0.52) * 40;
    value = Math.max(5400, Math.min(6100, value));
    data.push({
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      value: Math.round(value * 100) / 100,
    });
  }
  data[data.length - 1].value = 5842.31;
  return data;
};

// ==================== SCENARIO DATA ====================
export const shockVariables = [
  { id: '^GSPC', label: 'S&P 500', category: 'equity' },
  { id: 'CL=F', label: 'Crude Oil', category: 'commodities' },
  { id: 'DGS10', label: '10Y Treasury', category: 'rates' },
  { id: 'BAMLH0A0HYM2', label: 'Credit Spreads', category: 'fixed-income' },
  { id: '^VIX', label: 'VIX', category: 'volatility' },
  { id: 'FEDFUNDS', label: 'Fed Funds', category: 'rates' },
];

export const generateScenarioPaths = (numPaths = 20, horizon = 60): ScenarioResult[] => {
  const paths: number[][] = [];
  for (let p = 0; p < numPaths; p++) {
    const path = [0];
    let cumReturn = 0;
    for (let d = 1; d <= horizon; d++) {
      cumReturn += (Math.random() - 0.52) * 1.5;
      path.push(Math.round(cumReturn * 100) / 100);
    }
    paths.push(path);
  }
  const results: ScenarioResult[] = [];
  for (let d = 0; d <= horizon; d++) {
    const dayValues = paths.map(p => p[d]).sort((a, b) => a - b);
    results.push({
      day: d,
      median: dayValues[Math.floor(numPaths / 2)],
      p5: dayValues[Math.floor(numPaths * 0.05)],
      p95: dayValues[Math.floor(numPaths * 0.95)],
      paths: dayValues,
    });
  }
  return results;
};

export const returnDistribution = [
  { bucket: '-20%', freq: 2 }, { bucket: '-17.5%', freq: 3 }, { bucket: '-15%', freq: 5 },
  { bucket: '-12.5%', freq: 7 }, { bucket: '-10%', freq: 9 }, { bucket: '-7.5%', freq: 12 },
  { bucket: '-5%', freq: 15 }, { bucket: '-2.5%', freq: 18 }, { bucket: '0%', freq: 14 },
  { bucket: '2.5%', freq: 11 }, { bucket: '5%', freq: 8 }, { bucket: '7.5%', freq: 5 },
  { bucket: '10%', freq: 3 }, { bucket: '12.5%', freq: 2 }, { bucket: '15%', freq: 1 },
];

export const impactSummary = [
  { variable: '^GSPC', p5: -15.81, median: -2.00, p95: 12.59, mean: -1.55 },
  { variable: '^VIX', p5: -55.96, median: 2.19, p95: 71.74, mean: 2.73 },
  { variable: 'DGS10', p5: -80.96, median: -25.26, p95: 44.55, mean: -22.88 },
  { variable: 'CL=F', p5: -24.80, median: 9.39, p95: 39.81, mean: 8.43 },
  { variable: 'XLF', p5: -28.33, median: -1.92, p95: 18.54, mean: -3.52 },
  { variable: 'BAMLH0A0HYM2', p5: -167.83, median: 23.31, p95: 214.47, mean: 23.67 },
];

// ==================== STRESS TEST DATA ====================
export interface PortfolioHolding {
  asset: string;
  weight: number;
  amount: number;
  category: string;
}

export const portfolioPresets: Record<string, PortfolioHolding[]> = {
  Conservative: [
    { asset: 'S&P 500', weight: 30, amount: 300000, category: 'equity' },
    { asset: '20Y Treasury Bonds', weight: 30, amount: 300000, category: 'fixed-income' },
    { asset: 'Investment Grade Bonds', weight: 20, amount: 200000, category: 'fixed-income' },
    { asset: 'Gold', weight: 10, amount: 100000, category: 'commodities' },
    { asset: 'Emerging Markets', weight: 10, amount: 100000, category: 'equity' },
  ],
  Aggressive: [
    { asset: 'S&P 500', weight: 50, amount: 500000, category: 'equity' },
    { asset: 'Nasdaq 100', weight: 20, amount: 200000, category: 'equity' },
    { asset: 'Emerging Markets', weight: 15, amount: 150000, category: 'equity' },
    { asset: 'Crude Oil', weight: 10, amount: 100000, category: 'commodities' },
    { asset: 'High Yield Bonds', weight: 5, amount: 50000, category: 'fixed-income' },
  ],
  Balanced: [
    { asset: 'S&P 500', weight: 40, amount: 400000, category: 'equity' },
    { asset: '20Y Treasury Bonds', weight: 20, amount: 200000, category: 'fixed-income' },
    { asset: 'Investment Grade Bonds', weight: 15, amount: 150000, category: 'fixed-income' },
    { asset: 'Gold', weight: 10, amount: 100000, category: 'commodities' },
    { asset: 'Emerging Markets', weight: 10, amount: 100000, category: 'equity' },
    { asset: 'Real Estate', weight: 5, amount: 50000, category: 'equity' },
  ],
};

export const stressTestResults = {
  var95: -65810,
  var99: -81696,
  cvar95: -77829,
  maxDrawdown: -106725,
  lossProbabilities: { any: 41, over5: 14, over10: 1 },
  sectorRisk: [
    { sector: 'Equity', value: 45, color: '#3b82f6' },
    { sector: 'Fixed Income', value: 28, color: '#a855f7' },
    { sector: 'International', value: 18, color: '#06b6d4' },
    { sector: 'Commodity', value: 9, color: '#f59e0b' },
  ],
  holdingRisk: [
    { holding: 'S&P 500', risk: -56312 },
    { holding: '20Y Treasury', risk: -20739 },
    { holding: 'Emerging Markets', risk: -20637 },
    { holding: 'Gold', risk: -8560 },
    { holding: 'IG Bonds', risk: -8490 },
  ],
};

// ==================== DASHBOARD REGIME STRIP ====================
export const dashboardRegimeStrip = [
  { regime: 'Calm', start: 'Jan 2024', end: 'Apr 2024', pct: 12 },
  { regime: 'Normal', start: 'Apr 2024', end: 'Jul 2024', pct: 11 },
  { regime: 'Elevated', start: 'Jul 2024', end: 'Sep 2024', pct: 8 },
  { regime: 'Stressed', start: 'Sep 2024', end: 'Oct 2024', pct: 5 },
  { regime: 'Normal', start: 'Oct 2024', end: 'Jan 2025', pct: 11 },
  { regime: 'Calm', start: 'Jan 2025', end: 'May 2025', pct: 15 },
  { regime: 'Normal', start: 'May 2025', end: 'Aug 2025', pct: 11 },
  { regime: 'Elevated', start: 'Aug 2025', end: 'Mar 2026', pct: 22 },
  { regime: 'Crisis', start: 'Mar 2026', end: 'Mar 2026', pct: 5 },
];
