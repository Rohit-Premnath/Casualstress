-- Bring older databases in line with the runtime schema expected by the app.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS regulatory;

ALTER TABLE IF EXISTS models.scenarios
    ADD COLUMN IF NOT EXISTS event_type VARCHAR(50),
    ADD COLUMN IF NOT EXISTS anchor_variable VARCHAR(50);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'models'
          AND table_name = 'scenarios'
          AND column_name = 'regime_condition'
          AND data_type <> 'character varying'
    ) THEN
        ALTER TABLE models.scenarios
            ALTER COLUMN regime_condition TYPE VARCHAR(50)
            USING regime_condition::VARCHAR;
    END IF;
END $$;

UPDATE models.scenarios
SET event_type = COALESCE(
        event_type,
        CASE
            WHEN shock_variable IN ('market_crash', '^GSPC') THEN 'market_crash'
            WHEN shock_variable IN ('credit_crisis', 'BAMLH0A0HYM2') THEN 'credit_crisis'
            WHEN shock_variable IN ('rate_shock', 'DGS10') THEN 'rate_shock'
            WHEN shock_variable IN ('global_shock', 'CL=F') THEN 'global_shock'
            WHEN shock_variable IN ('volatility_shock', '^VIX') THEN 'volatility_shock'
            WHEN shock_variable = 'pandemic_exogenous' THEN 'pandemic_exogenous'
            ELSE NULL
        END
    ),
    anchor_variable = COALESCE(
        anchor_variable,
        CASE
            WHEN shock_variable = 'market_crash' THEN '^GSPC'
            WHEN shock_variable = 'credit_crisis' THEN 'BAMLH0A0HYM2'
            WHEN shock_variable = 'rate_shock' THEN 'DGS10'
            WHEN shock_variable = 'global_shock' THEN 'CL=F'
            WHEN shock_variable = 'volatility_shock' THEN '^VIX'
            WHEN shock_variable = 'pandemic_exogenous' THEN '^GSPC'
            ELSE shock_variable
        END
    );

UPDATE models.scenarios
SET shock_variable = anchor_variable
WHERE anchor_variable IS NOT NULL
  AND shock_variable IN (
      'market_crash',
      'credit_crisis',
      'rate_shock',
      'global_shock',
      'volatility_shock',
      'pandemic_exogenous'
  );

CREATE INDEX IF NOT EXISTS idx_scenarios_created_at
    ON models.scenarios (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_scenarios_event_type
    ON models.scenarios (event_type, created_at DESC);

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
