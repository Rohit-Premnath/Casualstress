"""
AI Financial Risk Advisor
============================
A conversational AI that combines ALL CausalStress capabilities into
one intelligent interface. Users ask questions in plain English,
the advisor decides which backend tools to call, runs the analysis,
and returns a conversational answer.

This is the feature that turns CausalStress into a $10M product.

Capabilities:
1.  Current market regime query
2.  Portfolio stress testing via natural language
3.  Scenario generation from plain English
4.  Causal path tracing and explanation
5.  Regulatory divergence explanation
6.  Portfolio comparison
7.  Risk decomposition and recommendations
8.  Regime history and transitions
9.  Variable relationship queries
10. Full report generation
"""

import os
import sys
import json
from datetime import datetime

from anthropic import Anthropic
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json
import numpy as np

load_dotenv()


# ============================================
# CLIENTS
# ============================================

def get_anthropic():
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


# ============================================
# TOOL FUNCTIONS (what the advisor can call)
# ============================================

def tool_get_current_regime():
    """Get the current market regime with full details."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT date, regime_name, probability
        FROM models.regimes
        ORDER BY date DESC
        LIMIT 1
    """)
    current = cursor.fetchone()

    # Get streak
    cursor.execute("""
        SELECT regime_name, COUNT(*) as streak
        FROM (
            SELECT date, regime_name,
                   ROW_NUMBER() OVER (ORDER BY date DESC) -
                   ROW_NUMBER() OVER (PARTITION BY regime_name ORDER BY date DESC) as grp
            FROM models.regimes
        ) sub
        WHERE grp = (
            SELECT ROW_NUMBER() OVER (ORDER BY date DESC) -
                   ROW_NUMBER() OVER (PARTITION BY regime_name ORDER BY date DESC)
            FROM models.regimes
            ORDER BY date DESC
            LIMIT 1
        )
        GROUP BY regime_name
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """)
    streak_row = cursor.fetchone()

    # Get regime stats
    cursor.execute("""
        SELECT regime_name, COUNT(*) as days,
               ROUND(COUNT(*)::numeric / (SELECT COUNT(*) FROM models.regimes) * 100, 1) as pct
        FROM models.regimes
        GROUP BY regime_name
        ORDER BY COUNT(*) DESC
    """)
    distribution = cursor.fetchall()

    cursor.close()
    conn.close()

    return json.dumps({
        "current_date": str(current[0]),
        "regime": current[1],
        "confidence": float(current[2]),
        "streak_days": streak_row[1] if streak_row else 0,
        "regime_distribution": [
            {"regime": r[0], "days": r[1], "percentage": float(r[2])}
            for r in distribution
        ],
    })


def tool_get_regime_history(period="1year"):
    """Get regime changes over a specified period."""
    conn = get_db()
    cursor = conn.cursor()

    if period == "6months":
        days = 126
    elif period == "1year":
        days = 252
    elif period == "2years":
        days = 504
    else:
        days = 252

    cursor.execute(f"""
        SELECT date, regime_name, probability
        FROM models.regimes
        ORDER BY date DESC
        LIMIT {days}
    """)
    rows = cursor.fetchall()

    # Find regime transitions
    transitions = []
    prev_regime = None
    for date, regime, prob in reversed(rows):
        if prev_regime and regime != prev_regime:
            transitions.append({
                "date": str(date),
                "from": prev_regime,
                "to": regime,
                "confidence": float(prob),
            })
        prev_regime = regime

    cursor.close()
    conn.close()

    return json.dumps({
        "period": period,
        "days_analyzed": len(rows),
        "current_regime": rows[0][1] if rows else "unknown",
        "transitions": transitions,
        "n_transitions": len(transitions),
    })


def tool_get_causal_links(variable=None, top_n=15):
    """Get causal relationships, optionally filtered by a specific variable."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT adjacency_matrix, confidence_scores
        FROM models.causal_graphs
        WHERE method = 'dynotears_lasso'
        ORDER BY created_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return json.dumps({"error": "No causal graph found"})

    adj = row[0]
    conf = row[1]

    edges = []
    for key, data in adj.items():
        cause, effect = key.split("->")
        if variable:
            if cause != variable and effect != variable:
                continue
        edges.append({
            "cause": cause,
            "effect": effect,
            "weight": round(data["weight"], 4),
            "lag": data.get("lag", 0),
            "confidence": round(conf.get(key, 0), 2),
            "direction": "outgoing" if cause == variable else "incoming" if effect == variable else "related",
        })

    edges.sort(key=lambda x: x["weight"], reverse=True)

    return json.dumps({
        "variable_filter": variable,
        "total_edges": len(edges),
        "top_edges": edges[:top_n],
    })


def tool_trace_causal_path(source, target):
    """Trace the causal path between two variables through the graph."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT adjacency_matrix
        FROM models.causal_graphs
        WHERE method = 'dynotears_lasso'
        ORDER BY created_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return json.dumps({"error": "No causal graph found"})

    adj = row[0]

    # Build adjacency list
    graph = {}
    weights = {}
    for key, data in adj.items():
        cause, effect = key.split("->")
        if cause not in graph:
            graph[cause] = []
        graph[cause].append(effect)
        weights[(cause, effect)] = data["weight"]

    # BFS to find shortest causal path
    from collections import deque
    queue = deque([(source, [source])])
    visited = {source}

    while queue:
        node, path = queue.popleft()
        if node == target:
            # Build path with weights
            path_details = []
            for i in range(len(path) - 1):
                w = weights.get((path[i], path[i+1]), 0)
                path_details.append({
                    "from": path[i],
                    "to": path[i+1],
                    "weight": round(w, 4),
                })
            return json.dumps({
                "source": source,
                "target": target,
                "path_found": True,
                "path_length": len(path) - 1,
                "path": path,
                "path_details": path_details,
            })

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return json.dumps({
        "source": source,
        "target": target,
        "path_found": False,
        "message": f"No causal path found from {source} to {target}",
    })


def tool_run_stress_test(portfolio_type="conservative", shock_variable="^GSPC", shock_magnitude=-3.0):
    """Run a stress test on a portfolio with a specific shock."""
    # Load the most recent scenario matching this shock
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, shock_variable, shock_magnitude, scenario_paths, plausibility_scores
        FROM models.scenarios
        WHERE shock_variable = %s
        ORDER BY created_at DESC
        LIMIT 1
    """, (shock_variable,))
    row = cursor.fetchone()

    if not row:
        cursor.close()
        conn.close()
        return json.dumps({"error": f"No scenarios found for {shock_variable}. Run scenario generator first."})

    scenario_paths = row[3]

    # Define portfolio
    portfolios = {
        "conservative": {
            "name": "Conservative Portfolio",
            "notional": 1_000_000,
            "holdings": {"S&P 500": 0.30, "20Y Treasury Bonds": 0.30,
                        "Investment Grade Bonds": 0.20, "Gold": 0.10, "Emerging Markets": 0.10},
        },
        "aggressive": {
            "name": "Aggressive Growth Portfolio",
            "notional": 1_000_000,
            "holdings": {"S&P 500": 0.25, "NASDAQ 100": 0.25, "Technology": 0.15,
                        "Emerging Markets": 0.15, "Energy": 0.10, "High Yield Bonds": 0.10},
        },
        "balanced": {
            "name": "Balanced Portfolio",
            "notional": 1_000_000,
            "holdings": {"S&P 500": 0.35, "20Y Treasury Bonds": 0.20,
                        "Investment Grade Bonds": 0.15, "Gold": 0.10,
                        "Emerging Markets": 0.10, "Real Estate": 0.10},
        },
    }

    portfolio = portfolios.get(portfolio_type, portfolios["conservative"])

    # Map assets to variables
    asset_map = {
        "S&P 500": "^GSPC", "NASDAQ 100": "^NDX", "Technology": "XLK",
        "Financials": "XLF", "Energy": "XLE", "Healthcare": "XLV",
        "Consumer Discretionary": "XLY", "Real Estate": "XLRE", "Utilities": "XLU",
        "Crude Oil": "CL=F", "Gold": "GC=F", "20Y Treasury Bonds": "TLT",
        "Investment Grade Bonds": "LQD", "High Yield Bonds": "HYG",
        "Emerging Markets": "EEM",
    }

    # Compute portfolio returns
    portfolio_returns = []
    for path_data in scenario_paths:
        scenario_data = path_data["data"]
        port_return = 0.0
        for asset, weight in portfolio["holdings"].items():
            var_code = asset_map.get(asset, "")
            if var_code in scenario_data:
                cum_return = sum(scenario_data[var_code])
                simple_return = np.exp(cum_return) - 1
                port_return += weight * simple_return
        portfolio_returns.append(port_return)

    returns_arr = np.array(portfolio_returns)
    notional = portfolio["notional"]

    var_95 = float(np.percentile(returns_arr, 5))
    var_99 = float(np.percentile(returns_arr, 1))
    cvar_95_mask = returns_arr <= var_95
    cvar_95 = float(returns_arr[cvar_95_mask].mean()) if cvar_95_mask.any() else var_95
    max_loss = float(np.min(returns_arr))
    prob_loss = float(np.mean(returns_arr < 0))
    prob_5pct = float(np.mean(returns_arr < -0.05))
    prob_10pct = float(np.mean(returns_arr < -0.10))

    cursor.close()
    conn.close()

    return json.dumps({
        "portfolio": portfolio["name"],
        "notional": notional,
        "holdings": portfolio["holdings"],
        "shock": f"{shock_variable} ({shock_magnitude:+.1f}σ)",
        "var_95": {"pct": round(var_95 * 100, 2), "dollar": round(var_95 * notional, 0)},
        "var_99": {"pct": round(var_99 * 100, 2), "dollar": round(var_99 * notional, 0)},
        "cvar_95": {"pct": round(cvar_95 * 100, 2), "dollar": round(cvar_95 * notional, 0)},
        "max_loss": {"pct": round(max_loss * 100, 2), "dollar": round(max_loss * notional, 0)},
        "probabilities": {
            "any_loss": round(prob_loss * 100, 1),
            "loss_over_5pct": round(prob_5pct * 100, 1),
            "loss_over_10pct": round(prob_10pct * 100, 1),
        },
        "n_scenarios": len(portfolio_returns),
    })


def tool_get_regulatory_divergences():
    """Get the latest regulatory divergence report."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT r.name, d.divergences, d.causal_explanations
        FROM regulatory.causal_difference_reports d
        JOIN regulatory.scenarios r ON d.regulatory_scenario_id = r.id
        ORDER BY d.created_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return json.dumps({"error": "No regulatory reports found. Run regulatory_engine.py first."})

    return json.dumps({
        "scenario": row[0],
        "n_divergences": len(row[1]) if row[1] else 0,
        "divergences": row[1][:5] if row[1] else [],
        "explanations": row[2][:3] if row[2] else [],
    })


def tool_get_system_stats():
    """Get overall CausalStress system statistics."""
    conn = get_db()
    cursor = conn.cursor()

    stats = {}

    cursor.execute("SELECT COUNT(*) FROM processed.time_series_data")
    stats["processed_records"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT variable_code) FROM processed.time_series_data")
    stats["total_variables"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM models.causal_graphs")
    stats["causal_graphs_computed"] = cursor.fetchone()[0]

    cursor.execute("""
        SELECT method, created_at
        FROM models.causal_graphs
        ORDER BY created_at DESC
        LIMIT 1
    """)
    latest_graph = cursor.fetchone()
    if latest_graph:
        stats["latest_graph_method"] = latest_graph[0]
        stats["latest_graph_date"] = str(latest_graph[1])

    cursor.execute("SELECT COUNT(*) FROM models.scenarios")
    stats["scenario_sets"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM models.regimes")
    stats["regime_classifications"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM app.stress_test_results")
    stats["stress_tests_run"] = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    return json.dumps(stats)


# ============================================
# TOOL DEFINITIONS FOR CLAUDE
# ============================================

TOOLS = [
    {
        "name": "get_current_regime",
        "description": "Get the current market regime (calm, normal, elevated, stressed, high_stress, or crisis) with confidence level, streak duration, and historical distribution. Call this when the user asks about current market conditions, risk level, or regime.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_regime_history",
        "description": "Get regime transition history over a specified period. Shows when the market shifted between regimes. Call this when the user asks about how regimes have changed, regime transitions, or market history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["6months", "1year", "2years"],
                    "description": "Time period to analyze",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_causal_links",
        "description": "Get causal relationships from the causal graph. Can show all top links or filter by a specific variable to see what it causes and what causes it. Call this when the user asks about causal relationships, what drives what, or connections between variables.",
        "input_schema": {
            "type": "object",
            "properties": {
                "variable": {
                    "type": "string",
                    "description": "Optional: specific variable code to filter by (e.g., ^GSPC, CL=F, DGS10, XLF, ^VIX, BAMLH0A0HYM2, CPIAUCSL, UNRATE, FEDFUNDS). Leave empty for top overall links.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top links to return (default 15)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "trace_causal_path",
        "description": "Trace the causal transmission path between two variables through the causal graph. Shows the chain of cause-and-effect from source to target. Call this when the user asks 'how does X affect Y' or 'what's the path from oil to bank stocks'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source variable code (e.g., CL=F for oil, ^VIX for volatility)",
                },
                "target": {
                    "type": "string",
                    "description": "Target variable code (e.g., XLF for bank stocks, UNRATE for unemployment)",
                },
            },
            "required": ["source", "target"],
        },
    },
    {
        "name": "run_stress_test",
        "description": "Run a portfolio stress test with a specific shock scenario. Returns VaR, CVaR, max loss, and loss probabilities. Call this when the user asks about portfolio risk, what would happen to their portfolio, or stress test results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "portfolio_type": {
                    "type": "string",
                    "enum": ["conservative", "aggressive", "balanced"],
                    "description": "Which portfolio preset to test",
                },
                "shock_variable": {
                    "type": "string",
                    "description": "Variable to shock (^GSPC for market crash, CL=F for oil, DGS10 for rates, BAMLH0A0HYM2 for credit)",
                },
                "shock_magnitude": {
                    "type": "number",
                    "description": "Shock size in standard deviations (e.g., -3.0 for crash, +3.0 for spike)",
                },
            },
            "required": ["portfolio_type"],
        },
    },
    {
        "name": "get_regulatory_divergences",
        "description": "Get the latest DFAST/CCAR regulatory compliance report showing where our causal model disagrees with the Federal Reserve's projections and why. Call this when the user asks about regulatory stress tests, DFAST, CCAR, or Fed scenarios.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_system_stats",
        "description": "Get overall CausalStress system statistics: total variables tracked, records processed, causal graphs computed, scenarios generated, stress tests run. Call this for general system overview questions.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# Map tool names to functions
TOOL_FUNCTIONS = {
    "get_current_regime": tool_get_current_regime,
    "get_regime_history": tool_get_regime_history,
    "get_causal_links": tool_get_causal_links,
    "trace_causal_path": tool_trace_causal_path,
    "run_stress_test": tool_run_stress_test,
    "get_regulatory_divergences": tool_get_regulatory_divergences,
    "get_system_stats": tool_get_system_stats,
}


# ============================================
# SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """You are the CausalStress AI Financial Risk Advisor — an expert AI assistant 
that helps financial advisors, portfolio managers, and risk analysts understand and manage 
financial risk using our causal stress testing platform.

You have access to a suite of financial analysis tools that let you:
- Check the current market regime and its history
- Explore causal relationships between 56+ financial variables
- Trace cause-and-effect chains through the economy
- Run portfolio stress tests against various crisis scenarios
- Analyze regulatory compliance (DFAST/CCAR) divergences

IMPORTANT CONTEXT:
- The system tracks 56+ macro-financial variables (equities, rates, credit spreads by rating, 
  bank lending behavior, funding stress, commodities, currencies)
- Causal relationships are discovered using DYNOTEARS and PCMCI algorithms (not correlations)
- 6 market regimes are detected: calm, normal, elevated, stressed, high_stress, crisis
- The current date is approximately March 2026

VARIABLE CODE REFERENCE:
Equities: ^GSPC (S&P 500), ^NDX (NASDAQ), ^RUT (Russell 2000), XLK (Tech), XLF (Financials), 
XLE (Energy), XLV (Healthcare), XLY (Consumer Disc), XLU (Utilities), XLRE (Real Estate)
Rates: DGS10 (10Y yield), DGS2 (2Y yield), FEDFUNDS (Fed rate), T10Y2Y (yield curve)
Credit: BAMLH0A0HYM2 (HY spread), BAMLH0A3HYC (CCC spread), BAMLC0A0CM (IG spread)
Macro: CPIAUCSL (CPI), UNRATE (unemployment), PAYEMS (payrolls), A191RL1Q225SBEA (GDP)
Volatility: ^VIX, ^VVIX, ^MOVE (bond vol)
Commodities: CL=F (oil), GC=F (gold)
Currencies: DX-Y.NYB (dollar index), EURUSD=X

GUIDELINES:
- Always use tools to get real data before answering — never make up numbers
- Explain financial concepts simply when the user seems non-technical
- Be specific with numbers and cite the data from tools
- When discussing risk, always mention both the downside AND any protective elements
- Proactively suggest related analyses the user might want to run
- If a question requires multiple tools, call them in sequence
- Format currency as $X,XXX and percentages with one decimal"""


# ============================================
# CONVERSATION ENGINE
# ============================================

class AdvisorChat:
    def __init__(self):
        self.client = get_anthropic()
        self.conversation_history = []

    def process_tool_call(self, tool_name, tool_input):
        """Execute a tool and return the result."""
        func = TOOL_FUNCTIONS.get(tool_name)
        if not func:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            if tool_input:
                result = func(**tool_input)
            else:
                result = func()
            return result
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})

    def chat(self, user_message):
        """Process a user message and return the advisor's response."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        # Initial API call
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=self.conversation_history,
        )

        # Handle tool use loop
        while response.stop_reason == "tool_use":
            # Find tool use blocks
            tool_uses = [block for block in response.content if block.type == "tool_use"]

            # Build assistant message with all content blocks
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_content,
            })

            # Execute all tools and build results
            tool_results = []
            for tool_use in tool_uses:
                print(f"  [Tool: {tool_use.name}({json.dumps(tool_use.input)[:80]}...)]")
                result = self.process_tool_call(tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                })

            self.conversation_history.append({
                "role": "user",
                "content": tool_results,
            })

            # Continue the conversation with tool results
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.conversation_history,
            )

        # Extract final text response
        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text += block.text

        self.conversation_history.append({
            "role": "assistant",
            "content": final_text,
        })

        return final_text


# ============================================
# INTERACTIVE CLI
# ============================================

def run_interactive():
    """Run the advisor in interactive CLI mode."""
    print("=" * 60)
    print("  CAUSALSTRESS AI FINANCIAL RISK ADVISOR")
    print("=" * 60)
    print()
    print("  Ask me anything about financial risk, market regimes,")
    print("  causal relationships, portfolio stress tests, or")
    print("  regulatory compliance.")
    print()
    print("  Example questions:")
    print("  • What's the current market regime?")
    print("  • How risky is my conservative portfolio right now?")
    print("  • How does oil price affect bank stocks?")
    print("  • What if there's a 2008-style crash?")
    print("  • Why does our model disagree with the Fed?")
    print("  • What causes inflation in the causal graph?")
    print("  • Show me system statistics")
    print()
    print("  Type 'quit' to exit.")
    print("=" * 60)

    advisor = AdvisorChat()

    while True:
        print()
        user_input = input("  You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n  Goodbye! Stay risk-aware.")
            break

        print()
        try:
            response = advisor.chat(user_input)
            print(f"  Advisor: {response}")
        except Exception as e:
            print(f"  Error: {str(e)}")


# ============================================
# DEMO MODE
# ============================================

def run_demo():
    """Run a scripted demo of the advisor's capabilities."""
    print("=" * 60)
    print("  CAUSALSTRESS AI ADVISOR — CAPABILITY DEMO")
    print("=" * 60)

    advisor = AdvisorChat()

    demo_questions = [
        "What's the current market regime and how long have we been in it?",
        "How does oil price causally affect inflation in our model?",
        "How risky is the conservative portfolio if the market crashes?",
        "What are the top causal drivers of the VIX?",
        "Does our model agree with the Fed's DFAST scenario?",
    ]

    for i, question in enumerate(demo_questions):
        print(f"\n\n{'#'*60}")
        print(f"  DEMO {i+1}: {question}")
        print(f"{'#'*60}")
        print(f"\n  You: {question}\n")

        try:
            response = advisor.chat(question)
            print(f"  Advisor: {response}")
        except Exception as e:
            print(f"  Error: {str(e)}")

    print(f"\n\n{'='*60}")
    print("✓ AI Advisor demo complete!")
    print("=" * 60)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CausalStress AI Financial Risk Advisor")
    parser.add_argument("--mode", choices=["chat", "demo"], default="chat",
                       help="chat=interactive conversation, demo=scripted capability demo")

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    else:
        run_interactive()