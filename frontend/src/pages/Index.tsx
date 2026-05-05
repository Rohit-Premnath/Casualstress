import { motion } from "framer-motion";
import { Activity, GitBranch, Network, FlaskConical, ShieldAlert, ArrowUp, Shield } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell } from "recharts";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.06 } },
};

const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

const legends = [
  { label: "Calm", color: "#10b981" },
  { label: "Normal", color: "#22d3ee" },
  { label: "Elevated", color: "#f59e0b" },
  { label: "Stressed", color: "#ef4444" },
  { label: "Crisis", color: "#991b1b" },
];

const regimeStepColors: Record<string, string> = {
  Calm: "#10b981",
  Normal: "#22d3ee",
  Elevated: "#f59e0b",
  Stressed: "#ef4444",
  Crisis: "#991b1b",
};

const regimeNameColor: Record<string, string> = {
  Calm: "text-emerald-400",
  Normal: "text-cyan-400",
  Elevated: "text-amber-400",
  Stressed: "text-red-400",
  Crisis: "text-red-600",
};

const regimeGlowClass: Record<string, string> = {
  Calm: "text-glow-green",
  Normal: "text-glow-cyan",
  Elevated: "text-glow-amber",
  Stressed: "text-glow-red",
  Crisis: "text-glow-red",
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload?.length) {
    return (
      <div className="glass px-3 py-2 text-xs rounded-lg">
        <p className="text-muted-foreground">{label}</p>
        <p className="font-mono font-semibold text-foreground">{payload[0].value.toLocaleString()}</p>
      </div>
    );
  }
  return null;
};

const RegimeChartTooltip = ({ active, payload }: any) => {
  if (active && payload?.length) {
    const dataPoint = payload[0]?.payload;
    return (
      <div className="glass px-3 py-2 text-xs rounded-lg">
        <p className="text-muted-foreground">{dataPoint?.month}</p>
        <div className="flex items-center gap-1.5 mt-1">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: regimeStepColors[dataPoint?.regime] || "#94a3b8" }} />
          <span className="font-semibold text-foreground">{dataPoint?.regime}</span>
        </div>
      </div>
    );
  }
  return null;
};

const SectionUnavailable = ({ message, compact = false }: { message: string; compact?: boolean }) => (
  <div
    className={`flex items-center justify-center rounded-xl border border-dashed border-border bg-secondary/20 px-4 text-center ${
      compact ? "min-h-[84px]" : "h-full min-h-[176px]"
    }`}
  >
    <p className="text-sm text-muted-foreground">{message}</p>
  </div>
);

const Index = () => {
  const summaryQuery = useQuery({
    queryKey: ["dashboard-summary"],
    queryFn: api.dashboard.getSummary,
    retry: 1,
    staleTime: 60_000,
  });

  const spxHistoryQuery = useQuery({
    queryKey: ["spx-history"],
    queryFn: () => api.dashboard.getSpxHistory(180),
    retry: 1,
    staleTime: 60_000,
  });

  const regimeChartQuery = useQuery({
    queryKey: ["regime-chart"],
    queryFn: () => api.dashboard.getRegimeChart(27),
    retry: 1,
    staleTime: 60_000,
  });

  const causalLinksQuery = useQuery({
    queryKey: ["top-causal-links"],
    queryFn: () => api.dashboard.getTopCausalLinks(10),
    retry: 1,
    staleTime: 60_000,
  });

  const summary = summaryQuery.data;
  const regime = summary?.currentRegime;
  const system = summary?.system;
  const spxValue = summary?.spx?.value;
  const spData = spxHistoryQuery.data ?? [];
  const regimeChartData = regimeChartQuery.data ?? [];
  const topLinks = causalLinksQuery.data ?? [];

  const todayIndex = regimeChartData.length - 2;
  const hasAnyData = [summary, spxHistoryQuery.data, regimeChartQuery.data, causalLinksQuery.data].some(Boolean);
  const isFullyLive = !!summary && !!spxHistoryQuery.data && !!regimeChartQuery.data && !!causalLinksQuery.data;
  const isAnyLoading = [summaryQuery, spxHistoryQuery, regimeChartQuery, causalLinksQuery].some((q) => q.isLoading);

  const regimeColor = regime ? (regimeNameColor[regime.name] ?? "text-amber-400") : "text-muted-foreground";
  const regimeDotColor = regime ? (regimeStepColors[regime.name] ?? "#f59e0b") : "#94a3b8";
  const regimeGlow = regime ? (regimeGlowClass[regime.name] ?? "") : "";
  const tradingDayRangeLabel = system?.startDate && system?.endDate
    ? `${new Date(system.startDate).getFullYear()} – ${new Date(system.endDate).getFullYear()}`
    : "Date range unavailable";

  return (
    <div className="p-6 md:p-8 max-w-[1440px] mx-auto space-y-6">
      {!isAnyLoading && (
        <div className={`flex items-center gap-1.5 text-[10px] font-mono ${isFullyLive ? "text-emerald-400" : hasAnyData ? "text-amber-400" : "text-muted-foreground"}`}>
          <div className={`w-1.5 h-1.5 rounded-full ${isFullyLive ? "bg-emerald-400 animate-pulse" : hasAnyData ? "bg-amber-400" : "bg-muted-foreground"}`} />
          {isFullyLive ? "LIVE — All dashboard sections connected" : hasAnyData ? "PARTIAL — Some dashboard sections unavailable" : "OFFLINE — Dashboard data unavailable"}
        </div>
      )}

      <motion.div variants={container} initial="hidden" animate="show" className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <motion.div variants={item} className="glass-hover rounded-2xl p-5 flex flex-col gap-3 group cursor-default">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground">Market Regime</span>
            <Activity className={`w-4 h-4 ${regimeColor} opacity-60 group-hover:opacity-100 transition-opacity`} />
          </div>
          {regime ? (
            <>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: regimeDotColor }} />
                <span className={`text-2xl font-bold font-mono ${regimeColor} ${regimeGlow}`}>{regime.name.toUpperCase()}</span>
              </div>
              <span className="text-[11px] text-muted-foreground">{regime.streak} days • {regime.confidence}% confidence</span>
            </>
          ) : (
            <SectionUnavailable compact message="Current regime unavailable." />
          )}
        </motion.div>

        <motion.div variants={item} className="glass-hover rounded-2xl p-5 flex flex-col gap-3 group cursor-default">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground">Variables Tracked</span>
            <Network className="w-4 h-4 text-primary opacity-60 group-hover:opacity-100 transition-opacity" />
          </div>
          {system ? (
            <>
              <span className="text-4xl font-mono font-light text-foreground">{system.variables}</span>
              <div className="flex items-center gap-1.5">
                <ArrowUp className="w-3 h-3 text-emerald-400" />
                <span className="text-[11px] text-emerald-400 font-medium">Active</span>
              </div>
            </>
          ) : (
            <SectionUnavailable compact message="Variable stats unavailable." />
          )}
        </motion.div>

        <motion.div variants={item} className="glass-hover rounded-2xl p-5 flex flex-col gap-3 group cursor-default">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground">Causal Edges</span>
            <GitBranch className="w-4 h-4 text-primary opacity-60 group-hover:opacity-100 transition-opacity" />
          </div>
          {system ? (
            <>
              <span className="text-4xl font-mono font-light text-foreground">{system.causalEdges.toLocaleString()}</span>
              <span className="text-[11px] text-muted-foreground">ensemble discovery</span>
            </>
          ) : (
            <SectionUnavailable compact message="Causal edge count unavailable." />
          )}
        </motion.div>

        <motion.div variants={item} className="glass-hover rounded-2xl p-5 flex flex-col gap-3 group cursor-default">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground">Trading Days</span>
            <FlaskConical className="w-4 h-4 text-primary opacity-60 group-hover:opacity-100 transition-opacity" />
          </div>
          {system ? (
            <>
              <span className="text-4xl font-mono font-light text-foreground">{system.tradingDays.toLocaleString()}</span>
              <span className="text-[11px] text-muted-foreground">{tradingDayRangeLabel}</span>
            </>
          ) : (
            <SectionUnavailable compact message="Trading-day coverage unavailable." />
          )}
        </motion.div>

        <motion.div variants={item} className="glass-hover rounded-2xl p-5 flex flex-col gap-3 group cursor-default">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground">S&amp;P 500</span>
            <ShieldAlert className="w-4 h-4 text-blue-400 opacity-60 group-hover:opacity-100 transition-opacity" />
          </div>
          {spxValue !== null && spxValue !== undefined ? (
            <>
              <span className="text-4xl font-mono font-light text-foreground">{spxValue.toLocaleString()}</span>
              <span className="text-[11px] text-muted-foreground">latest close</span>
            </>
          ) : (
            <SectionUnavailable compact message="Latest S&P 500 value unavailable." />
          )}
        </motion.div>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }} className="glass rounded-2xl p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Recent Regime History</h2>
          <div className="flex items-center gap-4">
            {legends.map((l) => (
              <div key={l.label} className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: l.color }} />
                <span className="text-[10px] text-muted-foreground">{l.label}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="h-40">
          {regimeChartData.length ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={regimeChartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <XAxis
                  dataKey="monthShort"
                  tick={{ fontSize: 9, fill: "currentColor", className: "text-muted-foreground" }}
                  axisLine={false}
                  tickLine={false}
                  interval={2}
                />
                <YAxis
                  domain={[0, 5.5]}
                  ticks={[1, 2, 3, 4, 5]}
                  tickFormatter={(v: number) => ["", "Calm", "Normal", "Elevated", "Stressed", "Crisis"][v] || ""}
                  tick={{ fontSize: 8, fill: "currentColor", className: "text-muted-foreground" }}
                  axisLine={false}
                  tickLine={false}
                  width={55}
                />
                <Tooltip content={<RegimeChartTooltip />} cursor={false} />
                {todayIndex >= 0 && (
                  <ReferenceLine
                    x={regimeChartData[todayIndex]?.monthShort}
                    stroke="currentColor"
                    strokeOpacity={0.4}
                    strokeDasharray="3 3"
                    label={{ value: "TODAY", position: "top", fontSize: 9, fill: "currentColor", className: "text-foreground font-mono" }}
                  />
                )}
                <Bar dataKey="value" radius={[3, 3, 0, 0]} barSize={16}>
                  {regimeChartData.map((entry: any, i: number) => (
                    <Cell key={i} fill={regimeStepColors[entry.regime] || "#666"} fillOpacity={0.75} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <SectionUnavailable message="Recent regime history unavailable." />
          )}
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.45 }} className="lg:col-span-3 glass rounded-2xl p-6">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-5">
            Top 10 Causal Links {topLinks.length > 0 && <span className="text-emerald-400 ml-2">● live</span>}
          </h2>
          {topLinks.length ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left text-[10px] uppercase tracking-widest text-muted-foreground pb-3 font-medium">Cause</th>
                  <th className="text-left text-[10px] uppercase tracking-widest text-muted-foreground pb-3 font-medium">Effect</th>
                  <th className="text-left text-[10px] uppercase tracking-widest text-muted-foreground pb-3 font-medium">Weight</th>
                  <th className="text-right text-[10px] uppercase tracking-widest text-muted-foreground pb-3 font-medium">Conf.</th>
                </tr>
              </thead>
              <tbody>
                {topLinks.map((link: any, i: number) => (
                  <tr key={i} className="border-b border-border/50 hover:bg-secondary/50 transition-colors">
                    <td className="py-2.5 font-mono text-xs text-foreground">{link.cause}</td>
                    <td className="py-2.5">
                      <span className="text-xs text-muted-foreground font-mono">→</span>
                      <span className="ml-1.5 font-mono text-xs text-foreground">{link.effect}</span>
                    </td>
                    <td className="py-2.5">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 rounded-full bg-muted overflow-hidden">
                          <div className="h-full rounded-full" style={{ width: `${Math.min((link.weight / 2.5) * 100, 100)}%`, background: "var(--btn-gradient)" }} />
                        </div>
                        <span className="font-mono text-xs text-muted-foreground w-8">{link.weight.toFixed(2)}</span>
                      </div>
                    </td>
                    <td className="py-2.5 text-right">
                      <span className="text-[10px] font-mono text-emerald-400">{link.confidence}%</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <SectionUnavailable message="Top causal links unavailable." />
          )}
        </motion.div>

        <div className="lg:col-span-2 space-y-6">
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }} className="glass rounded-2xl p-6">
            <div className="flex items-start justify-between mb-5">
              <div>
                <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">S&amp;P 500 Performance</h2>
                <div className="flex items-baseline gap-3 mt-2">
                  <span className="text-2xl font-bold font-mono text-foreground">
                    {spxValue !== null && spxValue !== undefined ? spxValue.toLocaleString() : "—"}
                  </span>
                </div>
              </div>
              <span className="text-[10px] text-muted-foreground font-mono">6M</span>
            </div>
            <div className="h-44">
              {spData.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={spData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="spGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
                        <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="date" tick={{ fontSize: 9, fill: "currentColor", className: "text-muted-foreground" }} axisLine={false} tickLine={false} interval={29} />
                    <YAxis domain={["dataMin - 50", "dataMax + 50"]} tick={{ fontSize: 9, fill: "currentColor", className: "text-muted-foreground" }} axisLine={false} tickLine={false} width={40} />
                    <Tooltip content={<CustomTooltip />} />
                    <Area type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={1.5} fill="url(#spGrad)" />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <SectionUnavailable message="S&P 500 history unavailable." />
              )}
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.55 }} className="glass rounded-2xl p-6">
            <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-4">Quick Actions</h2>
            <div className="flex gap-3">
              <Link to="/stress-test" className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-sm font-medium text-primary-foreground btn-gradient-blue transition-all duration-200 hover:scale-[1.02]">
                <Shield className="w-4 h-4" /> Run Stress Test
              </Link>
              <Link to="/causal-graph" className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-sm font-medium text-muted-foreground border border-border bg-secondary/30 hover:bg-secondary/60 hover:text-foreground transition-all duration-200 hover:scale-[1.02]">
                <GitBranch className="w-4 h-4" /> Explore Graph
              </Link>
              <Link to="/scenarios" className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-sm font-medium text-muted-foreground border border-border bg-secondary/30 hover:bg-secondary/60 hover:text-foreground transition-all duration-200 hover:scale-[1.02]">
                <FlaskConical className="w-4 h-4" /> View Scenarios
              </Link>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Index;
