-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

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
    event_type VARCHAR(50),
    anchor_variable VARCHAR(50),
    shock_variable VARCHAR(50),
    shock_magnitude DOUBLE PRECISION,
    regime_condition VARCHAR(50),
    causal_graph_id UUID REFERENCES models.causal_graphs(id),
    scenario_paths JSONB,
    plausibility_scores JSONB,
    n_scenarios INTEGER
);

CREATE INDEX IF NOT EXISTS idx_scenarios_created_at
    ON models.scenarios (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_scenarios_event_type
    ON models.scenarios (event_type, created_at DESC);

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

CREATE TABLE IF NOT EXISTS app.scheduler_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_name VARCHAR(100) NOT NULL,
    status VARCHAR(30) NOT NULL,
    details JSONB,
    executed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS app.regime_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL,
    old_regime VARCHAR(30),
    new_regime VARCHAR(30) NOT NULL,
    confidence DOUBLE PRECISION,
    notified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- SCHEMA: regulatory (regulatory scenario comparison)
-- ============================================
CREATE SCHEMA IF NOT EXISTS regulatory;

CREATE TABLE IF NOT EXISTS regulatory.scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    source VARCHAR(255) NOT NULL,
    year INTEGER NOT NULL,
    scenario_type VARCHAR(50),
    description TEXT,
    variables JSONB NOT NULL,
    horizon_quarters INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS regulatory.causal_difference_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    regulatory_scenario_id UUID REFERENCES regulatory.scenarios(id),
    portfolio JSONB,
    fed_projections JSONB NOT NULL,
    causal_projections JSONB NOT NULL,
    divergences JSONB NOT NULL,
    causal_explanations JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'CausalStress database initialized successfully!';
END $$;
