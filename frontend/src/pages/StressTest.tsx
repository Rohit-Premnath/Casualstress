import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Download, Info, TrendingDown, TrendingUp, ChevronDown, ChevronRight, ExternalLink, Plus, Trash2, Pencil, Check } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area } from "recharts";
import { portfolioPresets, categoryColors } from "@/data/mockData";
import { api, type ScenarioListItem, type StressTestResult } from "@/services/api";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

type ScenarioSource = "latest" | "saved";
type TemplateName = "Conservative" | "Balanced" | "Aggressive";

const templateNames: TemplateName[] = ["Conservative", "Balanced", "Aggressive"];

const riskTooltips: Record<string, string> = {
  "VaR (95%)": "Loss threshold exceeded in about 5% of stressed outcomes.",
  "VaR (99%)": "Loss threshold exceeded in about 1% of stressed outcomes.",
  "CVaR (95%)": "Average loss among the worst 5% of stressed outcomes.",
  "Worst Simulated Loss": "Most severe loss observed in the sampled scenario set.",
};

const parseCurrency = (value: string) => {
  const n = parseInt(value.replace(/[^0-9]/g, ""), 10);
  return Number.isNaN(n) ? 0 : n;
};

const formatScenarioDate = (value?: string | null) => {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, { month: "short", day: "numeric", year: "numeric" });
};

const formatScenarioOption = (scenario: ScenarioListItem) =>
  `${scenario.family} - ${scenario.severity} - ${scenario.horizon}d - ${formatScenarioDate(scenario.createdAt)}`;

const StressTest = () => {
  const navigate = useNavigate();
  const [scenarioSource, setScenarioSource] = useState<ScenarioSource>("latest");
  const [savedScenarios, setSavedScenarios] = useState<ScenarioListItem[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState("");
  const [activeTemplate, setActiveTemplate] = useState<TemplateName>("Conservative");
  const [isCustom, setIsCustom] = useState(false);
  const [customBase, setCustomBase] = useState<TemplateName>("Conservative");
  const [totalPortfolioValue, setTotalPortfolioValue] = useState(1_000_000);
  const [valueInput, setValueInput] = useState("1,000,000");
  const [customHoldings, setCustomHoldings] = useState(portfolioPresets.Conservative.map((holding) => ({ ...holding })));
  const [pageLoading, setPageLoading] = useState(true);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<StressTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hoveredMetric, setHoveredMetric] = useState<string | null>(null);
  const [comparisonOpen, setComparisonOpen] = useState(false);

  useEffect(() => {
    const loadScenarios = async () => {
      setPageLoading(true);
      try {
        const scenarios = await api.scenarios.list();
        setSavedScenarios(scenarios);
        if (scenarios.length) setSelectedScenarioId((current) => current || scenarios[0].id);
      } catch (e) {
        console.error(e);
        setError("Could not load saved scenarios from the live API.");
      } finally {
        setPageLoading(false);
      }
    };
    loadScenarios();
  }, []);

  const baseHoldings = portfolioPresets[isCustom ? customBase : activeTemplate] || portfolioPresets.Conservative;
  const templateHoldings = baseHoldings.map((holding) => ({ ...holding, amount: Math.round((totalPortfolioValue * holding.weight) / 100) }));
  const holdings = isCustom ? customHoldings : templateHoldings;
  const totalWeight = Math.round(holdings.reduce((sum, h) => sum + h.weight, 0) * 100) / 100;
  const totalAmount = holdings.reduce((sum, h) => sum + h.amount, 0);
  const isValid = Math.abs(totalWeight - 100) < 0.01 && Math.abs(totalAmount - totalPortfolioValue) < 1;
  const overUnder = totalAmount - totalPortfolioValue;
  const activeScenario = scenarioSource === "latest" ? savedScenarios[0] : savedScenarios.find((scenario) => scenario.id === selectedScenarioId) || savedScenarios[0];

  const resetResults = () => {
    setResults(null);
    setError(null);
  };

  const handleTemplateSelect = (template: TemplateName) => {
    setActiveTemplate(template);
    setIsCustom(false);
    resetResults();
  };

  const handleCustomize = () => {
    const source = portfolioPresets[activeTemplate] || portfolioPresets.Conservative;
    setCustomBase(activeTemplate);
    setCustomHoldings(source.map((holding) => ({ ...holding, amount: Math.round((totalPortfolioValue * holding.weight) / 100) })));
    setIsCustom(true);
    resetResults();
  };

  const handleWeightChange = (idx: number, newWeight: number) => {
    setCustomHoldings((prev) => prev.map((holding, i) => i === idx ? { ...holding, weight: newWeight, amount: Math.round((totalPortfolioValue * newWeight) / 100) } : holding));
  };

  const handleAmountChange = (idx: number, newAmount: number) => {
    setCustomHoldings((prev) => prev.map((holding, i) => i === idx ? { ...holding, amount: newAmount, weight: totalPortfolioValue > 0 ? Math.round(((newAmount / totalPortfolioValue) * 100) * 100) / 100 : 0 } : holding));
  };

  const handleRemoveHolding = (idx: number) => {
    setCustomHoldings((prev) => prev.filter((_, i) => i !== idx));
    resetResults();
  };

  const handleAddHolding = () => {
    setCustomHoldings((prev) => [...prev, { asset: "New Asset", weight: 0, amount: 0, category: "equity" }]);
    resetResults();
  };

  const handleValueCommit = () => {
    const parsed = parseCurrency(valueInput);
    const clamped = Math.max(1000, parsed);
    setTotalPortfolioValue(clamped);
    setValueInput(clamped.toLocaleString());
    if (isCustom) {
      setCustomHoldings((prev) => prev.map((holding) => ({ ...holding, amount: Math.round((clamped * holding.weight) / 100) })));
    }
    resetResults();
  };

  const handleRun = async () => {
    if (!isValid || !activeScenario?.id) return;
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const response = await api.stressTest.run(holdings.map((holding) => ({
        asset: holding.asset,
        weight: holding.weight,
        amount: holding.amount,
        category: holding.category,
      })), activeScenario.id);
      setResults(response);
    } catch (e) {
      console.error(e);
      setError("Could not run the live stress test. Check that the backend API and database are running.");
    } finally {
      setLoading(false);
    }
  };

  const riskMetrics = [
    { label: "VaR (95%)", value: results?.var95 ?? 0, pct: results ? Math.abs(results.var95 / results.portfolioValue) * 100 : 0, color: "hsl(var(--destructive))" },
    { label: "VaR (99%)", value: results?.var99 ?? 0, pct: results ? Math.abs(results.var99 / results.portfolioValue) * 100 : 0, color: "hsl(0 72% 40%)" },
    { label: "CVaR (95%)", value: results?.cvar95 ?? 0, pct: results ? Math.abs(results.cvar95 / results.portfolioValue) * 100 : 0, color: "hsl(25 95% 53%)" },
    { label: "Worst Simulated Loss", value: results?.maxDrawdown ?? 0, pct: results ? Math.abs(results.maxDrawdown / results.portfolioValue) * 100 : 0, color: "hsl(0 72% 35%)" },
  ];

  return (
    <div className="p-6 md:p-8 max-w-[1440px] mx-auto space-y-6">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="glass rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-primary" />
            <h2 className="text-xs uppercase tracking-widest text-muted-foreground font-medium">Scenario Set</h2>
          </div>
          <button onClick={() => navigate("/scenarios")} className="flex items-center gap-1.5 text-xs text-primary font-medium hover:underline transition-all">
            <ExternalLink className="w-3 h-3" /> Open Scenario Lab
          </button>
        </div>

        <div className="flex gap-2 mb-4">
          {[{ key: "latest" as ScenarioSource, label: "Latest Scenario" }, { key: "saved" as ScenarioSource, label: "Saved Scenarios" }].map((option) => (
            <button
              key={option.key}
              onClick={() => {
                setScenarioSource(option.key);
                resetResults();
              }}
              className={`px-4 py-2 rounded-xl text-sm transition-all ${
                scenarioSource === option.key
                  ? "bg-primary/15 text-primary border border-primary/30"
                  : "bg-secondary text-muted-foreground border border-border hover:text-foreground"
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>

        {scenarioSource === "saved" && (
          <div className="mb-4">
            <Select value={selectedScenarioId} onValueChange={(value) => { setSelectedScenarioId(value); resetResults(); }}>
              <SelectTrigger className="w-full max-w-md bg-secondary border-border">
                <SelectValue placeholder="Choose a saved scenario" />
              </SelectTrigger>
              <SelectContent>
                {savedScenarios.map((scenario) => (
                  <SelectItem key={scenario.id} value={scenario.id}>
                    {formatScenarioOption(scenario)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        <div className="flex flex-wrap gap-2 mb-3">
          {[
            { label: "Family", value: activeScenario?.family ?? "-" },
            { label: "Severity", value: activeScenario?.severity ?? "Severe" },
            { label: "Model", value: results?.scenario.model ?? "Full Model (Soft Filtered)" },
            { label: "ID", value: activeScenario?.id ?? "-" },
            { label: "Generated", value: formatScenarioDate(activeScenario?.createdAt) },
            { label: "Paths", value: String(activeScenario?.nScenarios ?? 0) },
            { label: "Horizon", value: `${activeScenario?.horizon ?? 0}d` },
          ].map((badge) => (
            <span key={badge.label} className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] bg-secondary text-secondary-foreground border border-border">
              <span className="text-muted-foreground">{badge.label}:</span>
              <span className="font-medium text-foreground">{badge.value}</span>
            </span>
          ))}
        </div>
        <p className="text-[11px] text-muted-foreground">Stress results are based on the selected crisis scenario set.</p>
        {pageLoading && <p className="text-[11px] text-muted-foreground mt-2">Loading saved scenarios...</p>}
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }} className="glass rounded-2xl p-6">
        <div className="flex items-center gap-2 mb-5">
          <Shield className="w-5 h-5 text-primary" />
          <h2 className="text-xl font-medium text-foreground">Portfolio Stress Test</h2>
        </div>

        <div className="mb-5 p-4 rounded-xl bg-secondary/50 border border-border">
          <label className="text-xs uppercase tracking-widest text-muted-foreground font-medium block mb-2">Total Portfolio Value</label>
          <div className="flex items-center gap-2">
            <span className="text-lg text-muted-foreground">$</span>
            <input
              type="text"
              value={valueInput}
              onChange={(e) => setValueInput(e.target.value)}
              onBlur={handleValueCommit}
              onKeyDown={(e) => e.key === "Enter" && handleValueCommit()}
              className="text-2xl font-mono font-medium bg-transparent border-none outline-none text-foreground w-full focus:ring-0"
              placeholder="1,000,000"
            />
          </div>
          <p className="text-[11px] text-muted-foreground mt-1.5">Total value used to calculate weights, dollar allocations, and stress losses.</p>
        </div>

        <div className="mb-2">
          <h3 className="text-xs uppercase tracking-widest text-muted-foreground font-medium mb-3">Portfolio Template</h3>
          <div className="flex items-center gap-2 flex-wrap">
            {templateNames.map((template) => (
              <button
                key={template}
                onClick={() => handleTemplateSelect(template)}
                className={`px-4 py-2 rounded-xl text-sm transition-all ${
                  !isCustom && activeTemplate === template
                    ? "bg-primary/15 text-primary border border-primary/30"
                    : "bg-secondary text-muted-foreground border border-border hover:text-foreground"
                }`}
              >
                {template}
              </button>
            ))}
            <div className="w-px h-6 bg-border mx-1" />
            <button
              onClick={handleCustomize}
              className={`px-4 py-2 rounded-xl text-sm transition-all flex items-center gap-1.5 ${
                isCustom
                  ? "bg-primary/15 text-primary border border-primary/30"
                  : "bg-secondary text-muted-foreground border border-border hover:text-foreground"
              }`}
            >
              <Pencil className="w-3 h-3" /> Customize Portfolio
            </button>
          </div>
        </div>

        {isCustom && (
          <div className="mt-3 mb-4 px-4 py-2.5 rounded-xl bg-primary/5 border border-primary/20 flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-primary">Custom Portfolio</span>
              <span className="text-xs text-muted-foreground ml-2">Based on {customBase}</span>
            </div>
            <button onClick={() => { setIsCustom(false); resetResults(); }} className="text-xs text-muted-foreground hover:text-foreground transition-colors">
              Reset to template
            </button>
          </div>
        )}

        <div className="mt-4 mb-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs uppercase tracking-widest text-muted-foreground font-medium">Portfolio Holdings</h3>
            {isCustom && (
              <button onClick={handleAddHolding} className="flex items-center gap-1 text-xs text-primary hover:underline">
                <Plus className="w-3 h-3" /> Add Holding
              </button>
            )}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left text-[10px] uppercase tracking-wider text-muted-foreground pb-2 font-medium">Asset</th>
                  <th className="text-left text-[10px] uppercase tracking-wider text-muted-foreground pb-2 font-medium">Weight</th>
                  <th className="text-right text-[10px] uppercase tracking-wider text-muted-foreground pb-2 font-medium">Dollar Amount</th>
                  {isCustom && <th className="w-10" />}
                </tr>
              </thead>
              <tbody>
                {holdings.map((holding, idx) => (
                  <tr key={`${holding.asset}-${idx}`} className="border-b border-border/50 group">
                    <td className="py-3 flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: categoryColors[holding.category] || "hsl(var(--muted-foreground))" }} />
                      <span className="text-foreground">{holding.asset}</span>
                    </td>
                    <td className="py-3">
                      {isCustom ? (
                        <div className="flex items-center gap-2">
                          <input
                            type="number"
                            value={holding.weight}
                            min={0}
                            max={100}
                            step={0.5}
                            onChange={(e) => handleWeightChange(idx, parseFloat(e.target.value) || 0)}
                            className="w-16 px-2 py-1 rounded-lg bg-secondary border border-border text-foreground font-mono text-xs text-right focus:outline-none focus:ring-1 focus:ring-primary"
                          />
                          <span className="text-xs text-muted-foreground">%</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-3">
                          <div className="w-20 h-1.5 rounded-full bg-muted overflow-hidden">
                            <div className="h-full rounded-full bg-primary" style={{ width: `${holding.weight}%` }} />
                          </div>
                          <span className="font-mono text-muted-foreground text-xs">{holding.weight}%</span>
                        </div>
                      )}
                    </td>
                    <td className="py-3 text-right">
                      {isCustom ? (
                        <input
                          type="number"
                          value={holding.amount}
                          min={0}
                          step={1000}
                          onChange={(e) => handleAmountChange(idx, parseInt(e.target.value) || 0)}
                          className="w-28 px-2 py-1 rounded-lg bg-secondary border border-border text-foreground font-mono text-xs text-right focus:outline-none focus:ring-1 focus:ring-primary"
                        />
                      ) : (
                        <span className="font-mono text-secondary-foreground">${holding.amount.toLocaleString()}</span>
                      )}
                    </td>
                    {isCustom && (
                      <td className="py-3 text-center">
                        <button onClick={() => handleRemoveHolding(idx)} className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive">
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="mb-5 p-4 rounded-xl border border-border bg-secondary/30">
          <div className="flex items-center gap-6 flex-wrap text-sm">
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Total Weight:</span>
              <span className={`font-mono text-xs px-2.5 py-0.5 rounded-full flex items-center gap-1 ${
                Math.abs(totalWeight - 100) < 0.01
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20"
                  : "bg-destructive/10 text-destructive border border-destructive/20"
              }`}>
                {Math.abs(totalWeight - 100) < 0.01 && <Check className="w-3 h-3" />}
                {totalWeight.toFixed(1)}% allocated
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Total Amount:</span>
              <span className="font-mono text-xs text-foreground">${totalAmount.toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Portfolio Value Match:</span>
              {isValid ? (
                <span className="font-mono text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
                  <Check className="w-3 h-3" /> Matches ${totalPortfolioValue.toLocaleString()}
                </span>
              ) : (
                <span className="font-mono text-xs px-2.5 py-0.5 rounded-full bg-destructive/10 text-destructive border border-destructive/20">
                  {overUnder > 0 ? "Over" : "Under"} by ${Math.abs(overUnder).toLocaleString()}
                </span>
              )}
            </div>
          </div>
          {!isValid && (
            <p className="text-xs text-destructive mt-3 flex items-center gap-1.5">
              <Info className="w-3 h-3" />
              Portfolio allocations must reconcile to the total portfolio value before running the stress test.
            </p>
          )}
        </div>

        <div className="flex items-center justify-end">
          <button
            onClick={handleRun}
            disabled={loading || !isValid || !activeScenario?.id}
            className="px-8 py-3 rounded-xl text-sm font-semibold text-primary-foreground btn-gradient-red transition-all duration-200 hover:scale-[1.02] flex items-center gap-2 disabled:opacity-60 disabled:hover:scale-100"
          >
            {loading ? <span className="animate-pulse">Running stress test...</span> : <><Shield className="w-4 h-4" /> Run Stress Test</>}
          </button>
        </div>
        {error && <p className="text-sm text-destructive mt-4">{error}</p>}
      </motion.div>

      <AnimatePresence>
        {results && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {riskMetrics.map((metric, idx) => (
                <motion.div
                  key={metric.label}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.08 }}
                  className="glass rounded-2xl p-5 relative group cursor-default"
                  style={{ borderTop: `2px solid ${metric.color}` }}
                  onMouseEnter={() => setHoveredMetric(metric.label)}
                  onMouseLeave={() => setHoveredMetric(null)}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase tracking-widest text-muted-foreground">{metric.label}</span>
                    <Info className="w-3 h-3 text-muted-foreground/50" />
                  </div>
                  <div className="text-3xl font-mono font-light mt-2" style={{ color: metric.color }}>
                    ${Math.abs(metric.value).toLocaleString()}
                  </div>
                  <span className="text-[11px] text-muted-foreground mt-1 block">{metric.pct.toFixed(1)}% of portfolio</span>
                  {hoveredMetric === metric.label && (
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 rounded-lg bg-card border border-border text-xs text-foreground shadow-lg z-10 w-56 text-center">
                      {riskTooltips[metric.label]}
                    </div>
                  )}
                </motion.div>
              ))}
            </div>

            <div className="flex flex-wrap gap-3">
              {[
                `${results.lossProbabilities.any}% chance of any loss`,
                `${results.lossProbabilities.over5}% chance of loss exceeding 5%`,
                `${results.lossProbabilities.over10}% chance of loss exceeding 10%`,
              ].map((label) => (
                <span key={label} className="px-4 py-2 rounded-full text-sm bg-destructive/10 text-destructive border border-destructive/20 font-mono">
                  {label}
                </span>
              ))}
            </div>

            <div className="glass rounded-2xl p-6">
              <h3 className="text-xs uppercase tracking-widest text-muted-foreground mb-1">Portfolio Loss Distribution</h3>
              <p className="text-[11px] text-muted-foreground mb-4">Weighted P&L outcomes across all stressed scenarios.</p>
              <div className="h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={results.pnlDistribution} margin={{ left: 0, right: 0 }}>
                    <defs>
                      <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="hsl(var(--destructive))" stopOpacity={0.35} />
                        <stop offset="100%" stopColor="hsl(var(--destructive))" stopOpacity={0.05} />
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="bucket" tick={{ fontSize: 10, fill: "currentColor", className: "text-muted-foreground" }} axisLine={false} tickLine={false} />
                    <YAxis hide />
                    <Tooltip
                      contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 11, color: "hsl(var(--foreground))" }}
                      formatter={(value: number) => [value, "Scenarios"]}
                    />
                    <Area type="monotone" dataKey="freq" stroke="hsl(var(--destructive))" fill="url(#pnlGrad)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass rounded-2xl p-6">
                <h3 className="text-xs uppercase tracking-widest text-muted-foreground mb-1">Sector Risk Attribution</h3>
                <p className="text-[11px] text-muted-foreground mb-4">Estimated downside contribution by asset category.</p>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={results.sectorRisk} dataKey="value" nameKey="sector" cx="50%" cy="50%" innerRadius={60} outerRadius={90}>
                        {results.sectorRisk.map((sector, idx) => (
                          <Cell key={idx} fill={sector.color} fillOpacity={0.8} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 11, color: "hsl(var(--foreground))" }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex items-center justify-center gap-4 mt-2 flex-wrap">
                  {results.sectorRisk.map((sector) => (
                    <div key={sector.sector} className="flex items-center gap-1.5">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: sector.color }} />
                      <span className="text-[10px] text-muted-foreground">{sector.sector}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="glass rounded-2xl p-6">
                <h3 className="text-xs uppercase tracking-widest text-muted-foreground mb-1">Risk by Holding</h3>
                <p className="text-[11px] text-muted-foreground mb-4">Estimated downside contribution by individual position.</p>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={results.holdingRisk} layout="vertical" margin={{ left: 80, right: 60 }}>
                      <XAxis type="number" tick={{ fontSize: 9, fill: "currentColor", className: "text-muted-foreground" }} axisLine={false} tickLine={false} />
                      <YAxis type="category" dataKey="holding" tick={{ fontSize: 10, fill: "currentColor", className: "text-muted-foreground" }} axisLine={false} tickLine={false} width={75} />
                      <Tooltip
                        contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 11, color: "hsl(var(--foreground))" }}
                        formatter={(value: number) => [`$${Math.abs(value).toLocaleString()}`, "Risk"]}
                      />
                      <Bar dataKey="risk" radius={[0, 4, 4, 0]}>
                        {results.holdingRisk.map((_, idx) => (
                          <Cell key={idx} fill="hsl(var(--destructive))" fillOpacity={0.7} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass rounded-2xl p-6">
                <div className="flex items-center gap-2 mb-4">
                  <TrendingDown className="w-4 h-4 text-destructive" />
                  <h3 className="text-xs uppercase tracking-widest text-muted-foreground">Top Risk Contributors</h3>
                </div>
                <div className="space-y-3">
                  {results.topContributors.map((contributor) => (
                    <div key={contributor.asset} className="flex items-center justify-between">
                      <span className="text-sm text-foreground">{contributor.asset}</span>
                      <div className="flex items-center gap-3">
                        <div className="w-24 h-1.5 rounded-full bg-muted overflow-hidden">
                          <div className="h-full rounded-full bg-destructive/70" style={{ width: `${Math.min(contributor.pct, 100)}%` }} />
                        </div>
                        <span className="font-mono text-xs text-destructive w-20 text-right">-${Math.abs(contributor.contribution).toLocaleString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="glass rounded-2xl p-6">
                <div className="flex items-center gap-2 mb-4">
                  <TrendingUp className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                  <h3 className="text-xs uppercase tracking-widest text-muted-foreground">Top Shock Absorbers</h3>
                </div>
                {results.topAbsorbers.length ? (
                  <div className="space-y-3">
                    {results.topAbsorbers.map((absorber) => (
                      <div key={absorber.asset} className="flex items-center justify-between">
                        <span className="text-sm text-foreground">{absorber.asset}</span>
                        <div className="flex items-center gap-3">
                          <div className="w-24 h-1.5 rounded-full bg-muted overflow-hidden">
                            <div className="h-full rounded-full bg-emerald-500/70" style={{ width: `${Math.min(absorber.pct, 100)}%` }} />
                          </div>
                          <span className="font-mono text-xs text-emerald-600 dark:text-emerald-400 w-20 text-right">+${absorber.contribution.toLocaleString()}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No positive shock absorbers were identified in this scenario set.</p>
                )}
              </div>
            </div>

            <div className="glass rounded-2xl p-5 border-l-2 border-primary/30">
              <h3 className="text-xs uppercase tracking-widest text-muted-foreground mb-2">How to Read This Stress Test</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                This portfolio is evaluated against the selected crisis scenario set. Losses are measured across many simulated paths.
                VaR and CVaR summarize downside severity at different confidence levels. Attribution charts show where the risk is concentrated
                and which holdings provide diversification benefit under stress.
              </p>
            </div>

            <Collapsible open={comparisonOpen} onOpenChange={setComparisonOpen}>
              <CollapsibleTrigger className="w-full">
                <div className="glass rounded-2xl p-4 flex items-center justify-between cursor-pointer hover:bg-secondary/30 transition-all">
                  <div className="flex items-center gap-2">
                    {comparisonOpen ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
                    <span className="text-sm text-foreground font-medium">Compare to Baseline Portfolio</span>
                  </div>
                  <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Placeholder</span>
                </div>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="glass rounded-2xl rounded-t-none p-6 border-t-0 -mt-3">
                  <p className="text-sm text-muted-foreground mb-4">Compare the current portfolio against a different preset to see how risk shifts.</p>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    {[{ label: "VaR Difference", value: "-" }, { label: "CVaR Difference", value: "-" }, { label: "Sector Exposure Delta", value: "-" }, { label: "Diversification", value: "-" }].map((stat) => (
                      <div key={stat.label} className="bg-muted/30 rounded-xl p-4 text-center">
                        <span className="text-[10px] uppercase tracking-wider text-muted-foreground block mb-1">{stat.label}</span>
                        <span className="font-mono text-lg text-foreground">{stat.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </CollapsibleContent>
            </Collapsible>

            <button className="w-full py-3 rounded-xl text-sm text-muted-foreground border border-border bg-secondary/30 hover:bg-secondary/60 transition-all flex items-center justify-center gap-2">
              <Download className="w-4 h-4" /> Export Report
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground/60 ml-1">(Coming Soon)</span>
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default StressTest;
