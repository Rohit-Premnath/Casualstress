-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================
-- SCHEMA: raw (staging area for ingested data)
-- ============================================
CREATE SCHEMA IF NOT EXISTS raw_fred;
CREATE SCHEMA IF NOT EXISTS raw_yahoo;

-- FRED raw data
CREATE TABLE IF NOT EXISTS raw_fred.observations (
    id BIGSERIAL PRIMARY KEY,
    series_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    value DOUBLE PRECISION,
    fetched_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(series_id, date)
);

-- Yahoo Finance raw data
CREATE TABLE IF NOT EXISTS raw_yahoo.daily_prices (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open_price DOUBLE PRECISION,
    high_price DOUBLE PRECISION,
    low_price DOUBLE PRECISION,
    close_price DOUBLE PRECISION,
    adj_close DOUBLE PRECISION,
    volume BIGINT,
    fetched_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, date)
);

-- ============================================
-- SCHEMA: processed (clean, aligned data)
-- ============================================
CREATE SCHEMA IF NOT EXISTS processed;

CREATE TABLE IF NOT EXISTS processed.time_series_data (
    date DATE NOT NULL,
    variable_code VARCHAR(50) NOT NULL,
    raw_value DOUBLE PRECISION,
    transformed_value DOUBLE PRECISION,
    source VARCHAR(20) NOT NULL,
    is_imputed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Convert to TimescaleDB hypertable for fast time-range queries
SELECT create_hypertable(
    'processed.time_series_data',
    'date',
    if_not_exists => TRUE
);

-- Create index for fast lookups
CREATE INDEX IF NOT EXISTS idx_ts_variable_date
    ON processed.time_series_data (variable_code, date DESC);

-- ============================================
-- SCHEMA: models (ML outputs)
-- ============================================
CREATE SCHEMA IF NOT EXISTS models;

CREATE TABLE IF NOT EXISTS models.causal_graphs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW(),
    method VARCHAR(20) NOT NULL,
    variables JSONB NOT NULL,
    adjacency_matrix JSONB NOT NULL,
    confidence_scores JSONB,
    structural_constraints JSONB,
    regime_id UUID,
    date_range_start DATE,
    date_range_end DATE
);

CREATE TABLE IF NOT EXISTS models.regimes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL,
    regime_label INTEGER NOT NULL,
    regime_name VARCHAR(30),
    probability DOUBLE PRECISION,
    transition_probs JSONB,
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_regimes_date
    ON models.regimes (date DESC);

CREATE TABLE IF NOT EXISTS models.scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW(),
    shock_variable VARCHAR(50),
    shock_magnitude DOUBLE PRECISION,
    regime_condition INTEGER,
    causal_graph_id UUID REFERENCES models.causal_graphs(id),
    scenario_paths JSONB,
    plausibility_scores JSONB,
    n_scenarios INTEGER
);

-- ============================================
-- SCHEMA: app (user-facing data)
-- ============================================
CREATE SCHEMA IF NOT EXISTS app;

CREATE TABLE IF NOT EXISTS app.stress_test_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio JSONB NOT NULL,
    scenario_id UUID REFERENCES models.scenarios(id),
    var_95 DOUBLE PRECISION,
    var_99 DOUBLE PRECISION,
    cvar_95 DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    sector_decomposition JSONB,
    marginal_contributions JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'CausalStress database initialized successfully!';
END $$;