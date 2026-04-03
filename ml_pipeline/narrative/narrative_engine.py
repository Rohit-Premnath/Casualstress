"""
LLM Narrative Engine
======================
Uses Claude (Anthropic API) to generate plain-English explanations
of stress test results, regulatory divergences, and portfolio risks.

Three capabilities:
1. Stress Test Narrator — turns VaR/CVaR numbers into advisor-ready language
2. Natural Language Scenario Input — parses "What if oil spikes and banks crash?"
   into shock parameters
3. Causal Difference Narrator — explains regulatory divergences in client-friendly terms

This is what turns CausalStress from a data tool into an advisory product.
Financial advisors don't speak in VaR — they need the STORY to tell clients.
"""

import os
import sys
import json
from datetime import datetime

from anthropic import Anthropic
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json

load_dotenv()


# ============================================
# ANTHROPIC CLIENT
# ============================================

def get_client():
    """Initialize Anthropic client."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_anthropic_api_key_here":
        print("ERROR: No Anthropic API key found!")
        print("Get one at: https://console.anthropic.com/settings/keys")
        print("Add to .env: ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    return Anthropic(api_key=api_key)


# ============================================
# DATABASE
# ============================================

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


# ============================================
# CAPABILITY 1: STRESS TEST NARRATOR
# ============================================

def narrate_stress_test(portfolio, metrics, sector_decomp, marginal_contrib, shock_info, regime_info=None):
    """
    Generate a plain-English narrative explanation of stress test results.
    This is what an advisor would hand to their client.
    """
    print("\nGenerating stress test narrative with Claude...")

    client = get_client()

    prompt = f"""You are a senior financial risk analyst writing a clear, professional 
stress test report for a financial advisor to share with their client. 

Write 3-4 paragraphs explaining these stress test results in plain English. 
Be specific with numbers but explain what they MEAN for the client's money.
Don't use jargon without explaining it. Be direct about risks but also mention 
any protective elements in the portfolio.

PORTFOLIO:
{json.dumps(portfolio, indent=2)}

SHOCK SCENARIO: {shock_info}

CURRENT MARKET REGIME: {regime_info or 'Elevated (above normal tension, not yet crisis)'}

RISK METRICS:
- VaR (95%): {metrics['var_95']['return']:.2%} (${metrics['var_95']['dollar']:,.0f})
  Meaning: {metrics['var_95']['description']}
- VaR (99%): {metrics['var_99']['return']:.2%} (${metrics['var_99']['dollar']:,.0f})
  Meaning: {metrics['var_99']['description']}
- CVaR (95%): {metrics['cvar_95']['return']:.2%} (${metrics['cvar_95']['dollar']:,.0f})
  Meaning: {metrics['cvar_95']['description']}
- Maximum Loss: {metrics['max_drawdown']['return']:.2%} (${metrics['max_drawdown']['dollar']:,.0f})
  Meaning: {metrics['max_drawdown']['description']}

LOSS PROBABILITIES:
- Probability of any loss: {metrics['probabilities']['prob_any_loss']:.0%}
- Probability of losing more than 5%: {metrics['probabilities']['prob_loss_5pct']:.0%}
- Probability of losing more than 10%: {metrics['probabilities']['prob_loss_10pct']:.0%}

SECTOR RISK BREAKDOWN:
{json.dumps(sector_decomp, indent=2)}

RISK BY INDIVIDUAL HOLDING (sorted worst to best):
{json.dumps(marginal_contrib, indent=2)}

Write the narrative now. Include:
1. A one-sentence executive summary of the overall risk level
2. What the biggest threat to this portfolio is and why
3. Which specific holdings are most vulnerable and which provide protection
4. A concrete recommendation for what the client could consider doing

Keep it under 400 words. Write as if speaking to an intelligent person who 
is NOT a finance expert. No bullet points — write in flowing paragraphs."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    narrative = message.content[0].text

    print(f"\n{'─'*60}")
    print("  STRESS TEST NARRATIVE (AI-Generated)")
    print(f"{'─'*60}")
    print(f"\n{narrative}")
    print(f"\n{'─'*60}")

    return narrative


# ============================================
# CAPABILITY 2: NATURAL LANGUAGE SCENARIO INPUT
# ============================================

def parse_scenario_input(user_input):
    """
    Parse a natural language scenario description into shock parameters.
    
    Example: "What if there's a banking crisis plus oil spike?"
    → shock_variables: [XLF, BAMLH0A0HYM2, CL=F]
    → magnitudes: [-3, +3, +3]
    """
    print(f"\nParsing natural language scenario: \"{user_input}\"")

    client = get_client()

    prompt = f"""You are a financial scenario parser. Convert the user's natural language 
crisis description into specific shock parameters for a financial stress testing system.

Available variables to shock (use these exact codes):
- ^GSPC: S&P 500 (overall US stock market)
- ^NDX: NASDAQ 100 (tech-heavy index)
- XLF: Financial sector ETF (banks, insurance)
- XLE: Energy sector ETF (oil companies)
- XLK: Technology sector ETF
- CL=F: Crude oil futures price
- GC=F: Gold futures price
- DGS10: 10-Year US Treasury yield
- DGS2: 2-Year US Treasury yield
- FEDFUNDS: Federal Funds Rate
- ^VIX: VIX volatility index (fear gauge)
- BAMLH0A0HYM2: High yield credit spread (corporate bond risk)
- CPIAUCSL: Consumer Price Index (inflation)
- UNRATE: Unemployment rate
- EURUSD=X: Euro/Dollar exchange rate
- EEM: Emerging markets ETF
- TLT: 20-Year Treasury bond ETF

Shock magnitudes are in standard deviations (sigma):
- +1σ to +2σ: moderate increase
- +3σ: severe increase  
- -1σ to -2σ: moderate decrease
- -3σ: severe decrease
- Direction depends on the variable (e.g., equity crash = negative, VIX spike = positive)

USER INPUT: "{user_input}"

Respond ONLY with a JSON object (no markdown, no explanation):
{{
    "scenario_name": "short descriptive name",
    "shocks": [
        {{"variable": "CODE", "magnitude": NUMBER, "description": "what this shock means"}},
        ...
    ],
    "regime_suggestion": "calm/normal/elevated/stressed/crisis",
    "horizon_days": 60
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text.strip()

    # Clean up any markdown formatting
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
    response_text = response_text.strip()

    try:
        parsed = json.loads(response_text)
        print(f"\n  Scenario: {parsed['scenario_name']}")
        print(f"  Regime: {parsed['regime_suggestion']}")
        print(f"  Horizon: {parsed['horizon_days']} days")
        print(f"  Shocks:")
        for shock in parsed["shocks"]:
            print(f"    {shock['variable']}: {shock['magnitude']:+.1f}σ — {shock['description']}")
        return parsed
    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse LLM response: {e}")
        print(f"  Raw response: {response_text}")
        return None


# ============================================
# CAPABILITY 3: CAUSAL DIFFERENCE NARRATOR
# ============================================

def narrate_regulatory_divergence(scenario_name, divergences, explanations):
    """
    Generate a rich narrative explanation of why our causal model
    disagrees with the Fed's regulatory scenario projections.
    """
    print(f"\nGenerating regulatory divergence narrative...")

    if not divergences:
        return "Our causal model's projections are broadly consistent with the Federal Reserve's scenario assumptions. No significant divergences were identified."

    client = get_client()

    prompt = f"""You are a senior regulatory risk analyst writing a memo to the Chief Risk Officer 
explaining where your bank's internal causal stress testing model disagrees with the 
Federal Reserve's DFAST scenario projections, and why these disagreements matter.

SCENARIO: {scenario_name}

DIVERGENCES FOUND (where our model differs >10% from Fed projections):
{json.dumps(divergences, indent=2)}

CAUSAL EXPLANATIONS (from our causal graph analysis):
{json.dumps(explanations, indent=2)}

Write a 3-4 paragraph executive memo that:
1. Summarizes the key disagreements in plain language
2. Explains the causal reasoning behind each major divergence
3. States the practical implication: does this mean the Fed is being too optimistic 
   or too pessimistic, and what should the bank prepare for?
4. Recommends specific actions the risk team should take

Use professional but clear language. A board member should understand this.
Reference specific numbers. Under 400 words. No bullet points."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    narrative = message.content[0].text

    print(f"\n{'─'*60}")
    print("  REGULATORY DIVERGENCE MEMO (AI-Generated)")
    print(f"{'─'*60}")
    print(f"\n{narrative}")
    print(f"\n{'─'*60}")

    return narrative


# ============================================
# CAPABILITY 4: PORTFOLIO RECOMMENDATION
# ============================================

def generate_recommendation(portfolio, metrics, marginal_contrib, regime_info="elevated"):
    """
    Generate specific portfolio adjustment recommendations based on
    stress test results and current market regime.
    """
    print(f"\nGenerating portfolio recommendations...")

    client = get_client()

    prompt = f"""You are a financial advisor analyzing a client's portfolio stress test results.
Based on the risk analysis, provide specific, actionable recommendations.

CURRENT MARKET REGIME: {regime_info}

PORTFOLIO:
{json.dumps(portfolio, indent=2)}

STRESS TEST RESULTS:
- VaR (95%): {metrics['var_95']['return']:.2%} (${metrics['var_95']['dollar']:,.0f})
- VaR (99%): {metrics['var_99']['return']:.2%} (${metrics['var_99']['dollar']:,.0f})
- CVaR (95%): {metrics['cvar_95']['return']:.2%} (${metrics['cvar_95']['dollar']:,.0f})
- Max Loss: {metrics['max_drawdown']['return']:.2%} (${metrics['max_drawdown']['dollar']:,.0f})
- Probability of any loss: {metrics['probabilities']['prob_any_loss']:.0%}
- Probability of >5% loss: {metrics['probabilities']['prob_loss_5pct']:.0%}
- Probability of >10% loss: {metrics['probabilities']['prob_loss_10pct']:.0%}

RISK BY HOLDING (worst to best):
{json.dumps(marginal_contrib, indent=2)}

Provide 3 specific, numbered recommendations. For each:
- State the exact action (e.g., "Reduce S&P 500 from 30% to 20%")
- Explain WHY this helps based on the stress test data
- Quantify the expected impact if possible

Keep it concise — under 250 words total. Write for an intelligent person, 
not a finance PhD."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    recommendation = message.content[0].text

    print(f"\n{'─'*60}")
    print("  PORTFOLIO RECOMMENDATIONS (AI-Generated)")
    print(f"{'─'*60}")
    print(f"\n{recommendation}")
    print(f"\n{'─'*60}")

    return recommendation


# ============================================
# DEMO: RUN ALL CAPABILITIES
# ============================================

def run_demo():
    """Demonstrate all narrative engine capabilities."""
    print("=" * 60)
    print("CAUSALSTRESS - LLM NARRATIVE ENGINE DEMO")
    print("=" * 60)

    # ── Demo 1: Stress Test Narrative ──
    print("\n\n" + "#" * 60)
    print("  DEMO 1: STRESS TEST NARRATIVE")
    print("#" * 60)

    sample_portfolio = {
        "name": "Conservative Portfolio",
        "notional": 1_000_000,
        "holdings": {
            "S&P 500": 0.30,
            "20Y Treasury Bonds": 0.30,
            "Investment Grade Bonds": 0.20,
            "Gold": 0.10,
            "Emerging Markets": 0.10,
        },
    }

    sample_metrics = {
        "var_95": {"return": -0.0658, "dollar": -65810, "description": "95% confident loss won't exceed $65,810"},
        "var_99": {"return": -0.0817, "dollar": -81696, "description": "99% confident loss won't exceed $81,696"},
        "cvar_95": {"return": -0.0778, "dollar": -77829, "description": "When losses exceed VaR95, average loss is $77,829"},
        "cvar_99": {"return": -0.1067, "dollar": -106725, "description": "When losses exceed VaR99, average loss is $106,725"},
        "max_drawdown": {"return": -0.1067, "dollar": -106725, "description": "Worst scenario loss: $106,725"},
        "probabilities": {"prob_any_loss": 0.60, "prob_loss_5pct": 0.14, "prob_loss_10pct": 0.01},
    }

    sample_sector = {
        "Equity": {"mean_contribution": -0.0213, "var_95_contribution": -0.0563, "dollar_at_risk": -56312, "weight": 0.30},
        "Fixed Income": {"mean_contribution": 0.0059, "var_95_contribution": -0.0164, "dollar_at_risk": -16368, "weight": 0.50},
        "International": {"mean_contribution": -0.0036, "var_95_contribution": -0.0206, "dollar_at_risk": -20637, "weight": 0.10},
        "Commodity": {"mean_contribution": 0.0032, "var_95_contribution": -0.0086, "dollar_at_risk": -8560, "weight": 0.10},
    }

    sample_marginal = {
        "S&P 500": {"weight": 0.30, "var_95_contribution": -0.0563, "dollar_at_risk": -56312, "worst_case": -84511},
        "20Y Treasury Bonds": {"weight": 0.30, "var_95_contribution": -0.0207, "dollar_at_risk": -20739, "worst_case": -32306},
        "Emerging Markets": {"weight": 0.10, "var_95_contribution": -0.0206, "dollar_at_risk": -20637, "worst_case": -29198},
        "Gold": {"weight": 0.10, "var_95_contribution": -0.0086, "dollar_at_risk": -8560, "worst_case": -13659},
        "Investment Grade Bonds": {"weight": 0.20, "var_95_contribution": -0.0085, "dollar_at_risk": -8490, "worst_case": -12322},
    }

    narrative = narrate_stress_test(
        sample_portfolio, sample_metrics, sample_sector,
        sample_marginal, "S&P 500 market crash (-3 sigma)",
        "Elevated (above normal tension, 230 day streak)"
    )

    # ── Demo 2: Natural Language Scenario Parsing ──
    print("\n\n" + "#" * 60)
    print("  DEMO 2: NATURAL LANGUAGE SCENARIO PARSING")
    print("#" * 60)

    test_scenarios = [
        "What if there's a banking crisis plus oil spike?",
        "Simulate a repeat of 2008 but with higher inflation",
        "What happens if China devalues the yuan and emerging markets crash?",
    ]

    for scenario_text in test_scenarios:
        parsed = parse_scenario_input(scenario_text)
        print()

    # ── Demo 3: Regulatory Divergence Memo ──
    print("\n\n" + "#" * 60)
    print("  DEMO 3: REGULATORY DIVERGENCE MEMO")
    print("#" * 60)

    # Load divergences from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT divergences, causal_explanations
        FROM regulatory.causal_difference_reports
        ORDER BY created_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row and row[0]:
        memo = narrate_regulatory_divergence(
            "DFAST 2026 Severely Adverse", row[0], row[1]
        )
    else:
        print("  No regulatory divergences found in database. Run regulatory_engine.py first.")

    # ── Demo 4: Portfolio Recommendations ──
    print("\n\n" + "#" * 60)
    print("  DEMO 4: PORTFOLIO RECOMMENDATIONS")
    print("#" * 60)

    recommendation = generate_recommendation(
        sample_portfolio, sample_metrics, sample_marginal, "elevated"
    )

    print("\n\n" + "=" * 60)
    print("✓ LLM Narrative Engine demo complete!")
    print("  4 capabilities demonstrated:")
    print("  1. Stress test narrative generation")
    print("  2. Natural language scenario parsing")
    print("  3. Regulatory divergence memo")
    print("  4. Portfolio recommendations")
    print("=" * 60)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    run_demo()