import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, ChevronDown, ChevronUp, Info } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Cell, Area, AreaChart, Line,
} from "recharts";

import { api, type ScenarioStressRangeRow } from "@/services/api";

type SeverityLevel = "Mild" | "Severe" | "Extreme";

const familyIcons: Record<string, string> = {
  "market-crash": "📉",
  "credit-crisis": "💳",
  "rate-shock": "📈",
  "global-shock": "🌍",
  "vol-shock": "⚡",
  pandemic: "🦠",
};

function formatMove(valueType: "return" | "level", val: number | null): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  if (valueType === "return") return `${val >= 0 ? "+" : ""}${val.toFixed(1)}%`;
  return `${val >= 0 ? "+" : ""}${val.toFixed(2)}`;
}

function formatCurrent(val: number | null): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  return Math.abs(val) >= 1000 ? val.toFixed(0) : val.toFixed(2);
}

function formatImpliedRange(row: ScenarioStressRangeRow): string {
  if (row.impliedLow === null || row.impliedHigh === null) return "—";
  return `${formatCurrent(row.impliedLow)} – ${formatCurrent(row.impliedHigh)}`;
}

const ScenarioLab = () => {
  const metadataQuery = useQuery({
    queryKey: ["scenario-metadata"],
    queryFn: api.scenarios.getMetadata,
    retry: 1,
    staleTime: 60_000,
  });

  const metadata = metadataQuery.data;
  const familyOptions = metadata?.families ?? [];
  const focusOptions = metadata?.focusVariables ?? [];
  const severityLevels = (metadata?.severityLevels as SeverityLevel[] | undefined) ?? ["Mild", "Severe", "Extreme"];
  const horizonOptions = metadata?.horizonOptions ?? [10, 30, 60];
  const displayedPaths = metadata?.displayedPaths ?? 200;
  const candidateCount = metadata?.candidateCount ?? 400;

  const [family, setFamily] = useState("market-crash");
  const [severity, setSeverity] = useState<SeverityLevel>("Severe");
  const [horizon, setHorizon] = useState(60);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [focusVar, setFocusVar] = useState<string>("spx");
  const [anchorVar, setAnchorVar] = useState("");
  const [anchorMag, setAnchorMag] = useState(-3);
  const [seed, setSeed] = useState(42);

  const scenarioMutation = useMutation({
    mutationFn: () => api.scenarios.generate({
      family_id: family,
      severity,
      horizon,
      displayed_paths: displayedPaths,
      anchor_variable_override: anchorVar || undefined,
      anchor_magnitude_override: anchorVar ? anchorMag : undefined,
      random_seed: seed,
    }),
  });

  const result = scenarioMutation.data;
  const effectiveFocusOptions = result?.focusVariables.length ? result.focusVariables : focusOptions;

  const selectedFocus = useMemo(() => {
    return effectiveFocusOptions.find((item) => item.id === focusVar) ?? effectiveFocusOptions[0] ?? null;
  }, [effectiveFocusOptions, focusVar]);

  const hasResults = !!result && !!selectedFocus;
  const focusSeries = selectedFocus ? result?.variables[selectedFocus.ticker] : undefined;
  const stressRows = result?.keyVariableStressRange ?? [];
  const displayedTemplate = result?.shockTemplate ?? [];

  return (
    <div className="p-6 md:p-8 max-w-[1440px] mx-auto">
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 min-h-[calc(100vh-4rem)]">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="lg:col-span-2 glass rounded-2xl p-6 flex flex-col gap-6">
          <h2 className="text-xl font-medium text-foreground">Design Your Crisis</h2>

          <div>
            <label className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2 block">Scenario Family</label>
            <div className="grid grid-cols-2 gap-2">
              {familyOptions.map((option) => (
                <button
                  key={option.id}
                  onClick={() => setFamily(option.id)}
                  className={`flex items-center gap-2 px-3 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 border ${
                    family === option.id
                      ? "bg-primary/10 border-primary text-foreground"
                      : "bg-secondary border-border text-muted-foreground hover:border-primary/30 hover:text-foreground"
                  }`}
                >
                  <span className="text-sm">{familyIcons[option.id] ?? "•"}</span>
                  {option.label}
                </button>
              ))}
            </div>
            {metadataQuery.isLoading && <p className="text-[10px] text-muted-foreground mt-2">Loading live scenario families...</p>}
          </div>

          <div>
            <label className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2 block">Severity</label>
            <div className="flex gap-2">
              {severityLevels.map((level) => (
                <button
                  key={level}
                  onClick={() => setSeverity(level)}
                  className={`flex-1 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 border ${
                    severity === level
                      ? "bg-primary/10 border-primary text-foreground"
                      : "bg-secondary border-border text-muted-foreground hover:border-primary/30"
                  }`}
                >
                  {level}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-muted-foreground mt-1.5">Controls the intensity of the multi-factor shock template.</p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2 block">Horizon</label>
              <div className="flex gap-1.5">
                {horizonOptions.map((option) => (
                  <button
                    key={option}
                    onClick={() => setHorizon(option)}
                    className={`flex-1 py-2 rounded-lg text-xs font-medium transition-all border ${
                      horizon === option
                        ? "bg-primary/10 border-primary text-foreground"
                        : "bg-secondary border-border text-muted-foreground hover:border-primary/30"
                    }`}
                  >
                    {option}d
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2 block">Displayed Paths</label>
              <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-secondary border border-border">
                <span className="text-xs font-mono text-foreground">{displayedPaths}</span>
              </div>
              <p className="text-[10px] text-muted-foreground mt-1">
                Engine generates {candidateCount} candidates internally and soft-weights the final {displayedPaths} scenarios.
              </p>
            </div>
          </div>

          <div className="border border-border rounded-xl overflow-hidden">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              Advanced Options
              {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="px-4 pb-4 space-y-3 border-t border-border pt-3">
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1 block">Anchor Variable Override</label>
                        <input
                          value={anchorVar}
                          onChange={(e) => setAnchorVar(e.target.value)}
                          placeholder="e.g. ^GSPC"
                          className="w-full bg-secondary border border-border rounded-lg px-3 py-2 text-xs text-foreground font-mono placeholder:text-muted-foreground/50 focus:ring-1 focus:ring-primary/30 outline-none"
                        />
                      </div>
                      <div>
                        <label className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1 block">Anchor Magnitude Override</label>
                        <input
                          type="number"
                          value={anchorMag}
                          onChange={(e) => setAnchorMag(+e.target.value)}
                          step={0.5}
                          className="w-full bg-secondary border border-border rounded-lg px-3 py-2 text-xs text-foreground font-mono focus:ring-1 focus:ring-primary/30 outline-none"
                        />
                      </div>
                    </div>
                    <div>
                      <label className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1 block">Random Seed</label>
                      <input
                        type="number"
                        value={seed}
                        onChange={(e) => setSeed(+e.target.value)}
                        className="w-full bg-secondary border border-border rounded-lg px-3 py-2 text-xs text-foreground font-mono focus:ring-1 focus:ring-primary/30 outline-none"
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] uppercase tracking-widest text-muted-foreground">Canonical model only</span>
                      <span className="text-[10px] text-muted-foreground/70">Live metadata-backed</span>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <button
            onClick={() => scenarioMutation.mutate()}
            disabled={scenarioMutation.isPending || metadataQuery.isLoading || !familyOptions.length}
            className="w-full py-4 rounded-xl text-sm font-semibold text-primary-foreground btn-gradient-blue transition-all duration-200 hover:scale-[1.02] flex items-center justify-center gap-2 mt-auto disabled:opacity-60"
          >
            {scenarioMutation.isPending ? (
              <span className="flex items-center gap-2">
                <span className="animate-pulse">●</span> Generating scenarios...
              </span>
            ) : (
              <><Zap className="w-4 h-4" /> Generate Scenarios</>
            )}
          </button>
          {scenarioMutation.isError && (
            <p className="text-[11px] text-destructive">
              Could not generate a live scenario set. Check that the backend API and database are running.
            </p>
          )}
          {metadataQuery.isError && (
            <p className="text-[11px] text-destructive">
              Could not load live scenario metadata. The scenario page is intentionally not falling back to hardcoded family definitions.
            </p>
          )}
        </motion.div>

        <div className="lg:col-span-3 space-y-6">
          <AnimatePresence>
            {hasResults && selectedFocus && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-6">
                <div className="flex items-center justify-between flex-wrap gap-3">
                  <h2 className="text-lg font-medium text-foreground">Scenario Results</h2>
                  <span className="text-[10px] font-mono px-3 py-1 rounded-full bg-accent/10 text-accent border border-accent/20">
                    {result.avgPlausibility.toFixed(2)} avg plausibility
                  </span>
                </div>

                <div className="flex flex-wrap gap-2">
                  {[
                    { label: "Model", value: result.model },
                    { label: "Family", value: result.family.label },
                    { label: "Graph", value: result.graph },
                    { label: "Filter", value: result.filter },
                    { label: "Candidates", value: `${result.candidateCount} → ${result.scenarioCount}` },
                  ].map((pill) => (
                    <span key={pill.label} className="text-[10px] px-3 py-1.5 rounded-lg bg-secondary border border-border text-muted-foreground">
                      <span className="text-foreground font-medium">{pill.label}:</span> {pill.value}
                    </span>
                  ))}
                </div>
                <p className="text-[10px] font-mono text-muted-foreground tracking-wide">
                  multi-root · stress-regime trained · causal propagation · soft-filtered
                </p>

                <div>
                  <label className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2 block">Focus Variable</label>
                  <div className="flex flex-wrap gap-1.5">
                    {effectiveFocusOptions.map((option) => (
                      <button
                        key={option.id}
                        onClick={() => setFocusVar(option.id)}
                        className={`px-3 py-1.5 rounded-lg text-[11px] font-medium transition-all border ${
                          focusVar === option.id
                            ? "bg-primary/10 text-foreground border-primary/40"
                            : "text-muted-foreground border-transparent hover:text-foreground hover:bg-secondary"
                        }`}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-1">Changes the visual emphasis of results. Does not affect scenario generation.</p>
                </div>

                <div className="glass rounded-2xl p-5">
                  <h3 className="text-xs uppercase tracking-widest text-muted-foreground mb-4">Shock Template Used</h3>
                  <div className="space-y-1">
                    {displayedTemplate.map((item) => {
                      const isHighlighted = item.ticker === selectedFocus.ticker;
                      return (
                        <div key={item.ticker} className={`flex items-center justify-between py-2 px-3 rounded-lg border-b border-border/30 last:border-0 transition-colors ${isHighlighted ? "bg-primary/5" : ""}`}>
                          <div className="flex items-center gap-3">
                            <span className={`font-mono text-xs w-28 ${isHighlighted ? "text-foreground font-semibold" : "text-foreground"}`}>{item.label}</span>
                            <span className="font-mono text-[10px] text-muted-foreground">{item.ticker}</span>
                          </div>
                          <span className={`font-mono text-sm font-semibold ${item.shock < 0 ? "text-destructive" : "text-accent"}`}>
                            {item.shock > 0 ? "+" : ""}
                            {item.shock}σ
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="glass rounded-2xl p-5">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xs uppercase tracking-widest text-muted-foreground">Return Distribution</h3>
                    <span className="text-[10px] font-mono text-muted-foreground px-2 py-0.5 rounded bg-secondary border border-border">{selectedFocus.label}</span>
                  </div>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={focusSeries?.distribution ?? []} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                        <XAxis dataKey="bucket" tickFormatter={(value) => `${value}`} tick={{ fontSize: 9, fill: "currentColor", className: "text-muted-foreground" }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fontSize: 9, fill: "currentColor", className: "text-muted-foreground" }} axisLine={false} tickLine={false} />
                        <ReferenceLine x={0} stroke="currentColor" strokeOpacity={0.2} strokeDasharray="4 4" />
                        <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 11, color: "hsl(var(--foreground))" }} />
                        <Bar dataKey="freq" radius={[4, 4, 0, 0]}>
                          {(focusSeries?.distribution ?? []).map((entry, index) => (
                            <Cell key={index} fill={entry.bucket < 0 ? "hsl(var(--destructive))" : "hsl(var(--primary))"} fillOpacity={0.7} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="glass rounded-2xl p-5">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xs uppercase tracking-widest text-muted-foreground">Weighted Scenario Paths</h3>
                    <span className="text-[10px] font-mono text-muted-foreground px-2 py-0.5 rounded bg-secondary border border-border">{selectedFocus.label}</span>
                  </div>
                  <div className="h-52">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={focusSeries?.fanChart ?? []} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                        <defs>
                          <linearGradient id="coneGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity={0.12} />
                            <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity={0.02} />
                          </linearGradient>
                        </defs>
                        <XAxis dataKey="day" tick={{ fontSize: 9, fill: "currentColor", className: "text-muted-foreground" }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fontSize: 9, fill: "currentColor", className: "text-muted-foreground" }} axisLine={false} tickLine={false} />
                        <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 11, color: "hsl(var(--foreground))" }} />
                        <Area type="monotone" dataKey="p95" stroke="none" fill="url(#coneGrad)" />
                        <Area type="monotone" dataKey="p5" stroke="none" fill="transparent" />
                        <Line type="monotone" dataKey="p95" stroke="hsl(var(--accent))" strokeWidth={1} strokeDasharray="4 4" dot={false} />
                        <Line type="monotone" dataKey="p5" stroke="hsl(var(--destructive))" strokeWidth={1} strokeDasharray="4 4" dot={false} />
                        <Line type="monotone" dataKey="median" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex items-center gap-4 mt-3 text-[10px] text-muted-foreground">
                    <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 rounded-full inline-block bg-primary" /> Weighted Median</span>
                    <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 rounded-full inline-block bg-destructive opacity-60" /> 5th Percentile</span>
                    <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 rounded-full inline-block bg-accent opacity-60" /> 95th Percentile</span>
                  </div>
                </div>

                <div className="glass rounded-2xl p-5">
                  <h3 className="text-xs uppercase tracking-widest text-muted-foreground mb-4">Key Variable Stress Range</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-border">
                          {["Variable", "Current", "5th Pctl Move", "Median Move", "95th Pctl Move", "Implied Median", "Implied Range"].map((heading) => (
                            <th key={heading} className="text-left text-[10px] uppercase tracking-wider text-muted-foreground pb-2 font-medium pr-3 whitespace-nowrap">{heading}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {stressRows.map((row) => {
                          const isHighlighted = row.ticker === selectedFocus.ticker;
                          return (
                            <tr key={row.variable} className={`border-b border-border/30 transition-colors ${isHighlighted ? "bg-primary/5" : ""}`}>
                              <td className={`py-2.5 font-mono pr-3 ${isHighlighted ? "text-foreground font-semibold" : "text-foreground"}`}>{row.variable}</td>
                              <td className="py-2.5 font-mono text-muted-foreground pr-3">{formatCurrent(row.current)}</td>
                              <td className={`py-2.5 font-mono pr-3 ${(row.p5Move ?? 0) < 0 ? "text-destructive" : "text-accent"}`}>{formatMove(row.valueType, row.p5Move)}</td>
                              <td className={`py-2.5 font-mono pr-3 ${(row.medianMove ?? 0) < 0 ? "text-destructive" : "text-accent"}`}>{formatMove(row.valueType, row.medianMove)}</td>
                              <td className={`py-2.5 font-mono pr-3 ${(row.p95Move ?? 0) < 0 ? "text-destructive" : "text-accent"}`}>{formatMove(row.valueType, row.p95Move)}</td>
                              <td className="py-2.5 font-mono text-foreground pr-3">{formatCurrent(row.impliedMedian)}</td>
                              <td className="py-2.5 font-mono text-muted-foreground whitespace-nowrap">{formatImpliedRange(row)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="glass rounded-2xl p-5 flex gap-3">
                  <Info className="w-4 h-4 text-muted-foreground shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-medium text-foreground mb-1">Why This Scenario Looks Like This</h4>
                    <p className="text-[11px] text-muted-foreground leading-relaxed">
                      This scenario uses stress-regime training, multi-root crisis templates ({result.family.label}), and causal propagation through market linkages. Outputs are soft-filtered by plausibility rather than hard-ranked. The engine generates {result.candidateCount} candidate paths and retains {result.scenarioCount} weighted scenarios based on structural coherence.
                      {result.severity === "Extreme" ? " Extreme severity amplifies the shock template materially, pushing tail outcomes further." : ""}
                    </p>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {!hasResults && !scenarioMutation.isPending && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-muted-foreground">
                <FlaskIcon className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p className="text-sm">Configure a crisis scenario and generate results</p>
                <p className="text-[11px] mt-1 opacity-60">Select a scenario family and severity to begin</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const FlaskIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className={className}>
    <path d="M9 3h6M10 3v7.5L4 18.5C3 20 4 22 6 22h12c2 0 3-2 2-3.5L14 10.5V3" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

export default ScenarioLab;
