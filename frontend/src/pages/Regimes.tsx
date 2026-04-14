import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import { useQueries } from "@tanstack/react-query";
import { api } from "@/services/api";
import { regimeColors, crisisAnnotations } from "@/data/mockData";

const Regimes = () => {
  const [hoveredSeg, setHoveredSeg] = useState<number | null>(null);

  const [currentQuery, timelineQuery, characteristicsQuery, transitionQuery] = useQueries({
    queries: [
      {
        queryKey: ["regimes-current"],
        queryFn: api.regimes.getCurrent,
        retry: 1,
        staleTime: 60_000,
      },
      {
        queryKey: ["regimes-timeline"],
        queryFn: api.regimes.getTimeline,
        retry: 1,
        staleTime: 60_000,
      },
      {
        queryKey: ["regimes-characteristics"],
        queryFn: api.regimes.getCharacteristics,
        retry: 1,
        staleTime: 60_000,
      },
      {
        queryKey: ["regimes-transition"],
        queryFn: api.regimes.getTransitionMatrix,
        retry: 1,
        staleTime: 60_000,
      },
    ],
  });

  const isLoading = [currentQuery, timelineQuery, characteristicsQuery, transitionQuery].some((q) => q.isLoading);
  const hasError = [currentQuery, timelineQuery, characteristicsQuery, transitionQuery].some((q) => q.isError);

  const currentRegime = currentQuery.data;
  const timeline = timelineQuery.data;
  const transitionMatrix = transitionQuery.data;
  const characteristics = useMemo(() => {
    if (!characteristicsQuery.data || !transitionMatrix?.labels?.length) return characteristicsQuery.data;
    const order = transitionMatrix.labels;
    return [...characteristicsQuery.data].sort(
      (a, b) => order.indexOf(a.regime) - order.indexOf(b.regime),
    );
  }, [characteristicsQuery.data, transitionMatrix]);

  const probabilities = useMemo(() => {
    if (!currentRegime) return {};
    const allLabels = transitionMatrix?.labels?.length
      ? transitionMatrix.labels
      : ["Calm", "Normal", "Elevated", "Stressed", "High Stress", "Crisis"];
    return Object.fromEntries(
      allLabels.map((label) => [label, label === currentRegime.name ? currentRegime.confidence : 0]),
    );
  }, [currentRegime, transitionMatrix]);

  const totalMonths = timeline?.reduce((a, s) => a + s.months, 0) ?? 0;
  const hoveredSegment = hoveredSeg !== null ? timeline?.[hoveredSeg] : null;
  const hoveredSegmentLeft = useMemo(() => {
    if (hoveredSeg === null || !timeline?.length || totalMonths === 0) return 0;
    const monthsBefore = timeline.slice(0, hoveredSeg).reduce((sum, seg) => sum + seg.months, 0);
    const currentWidth = timeline[hoveredSeg].months;
    return ((monthsBefore + currentWidth / 2) / totalMonths) * 100;
  }, [hoveredSeg, timeline, totalMonths]);

  if (isLoading) {
    return (
      <div className="p-6 md:p-8 max-w-[1440px] mx-auto space-y-6">
        <div className="glass rounded-2xl p-8 text-sm text-muted-foreground">Loading live regime data...</div>
      </div>
    );
  }

  if (
    hasError ||
    !currentRegime ||
    !timeline?.length ||
    !characteristics?.length ||
    !transitionMatrix?.labels?.length ||
    !transitionMatrix?.data?.length
  ) {
    return (
      <div className="p-6 md:p-8 max-w-[1440px] mx-auto space-y-6">
        <div className="glass rounded-2xl p-8">
          <h2 className="text-lg font-semibold text-foreground mb-2">Live regime data unavailable</h2>
          <p className="text-sm text-muted-foreground">
            This page does not use mock data. If you see this state, one or more live regime endpoints failed or returned empty results.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 md:p-8 max-w-[1440px] mx-auto space-y-8">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center py-8">
        <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground mb-3">Current Market Regime</p>
        <div className="flex items-center justify-center gap-3">
          <div
            className="w-3 h-3 rounded-full animate-pulse"
            style={{ backgroundColor: regimeColors[currentRegime.name] || "#f59e0b" }}
          />
          <h1
            className="text-6xl font-light font-mono"
            style={{ color: regimeColors[currentRegime.name] || "#f59e0b" }}
          >
            {currentRegime.name.toUpperCase()}
          </h1>
        </div>
        <p className="text-muted-foreground mt-3 text-sm">
          {currentRegime.confidence}% confidence • {currentRegime.streak} day streak
        </p>

        <div className="max-w-lg mx-auto mt-8 space-y-2">
          {Object.entries(probabilities).map(([regime, pct]) => (
            <div key={regime} className="flex items-center gap-3">
              <span className="text-xs text-muted-foreground w-20 text-right">{regime}</span>
              <div className="flex-1 h-5 rounded-full bg-secondary overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700"
                  style={{
                    width: `${Math.max(pct, 0.5)}%`,
                    backgroundColor: regimeColors[regime] || "#666",
                  }}
                />
              </div>
              <span className="font-mono text-xs text-muted-foreground w-12">{pct.toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="glass rounded-2xl p-6"
      >
        <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-5">Regime History 2005–2026</h2>
        <div className="relative">
          {hoveredSegment && (
            <div
              className="absolute top-0 z-20 min-w-[190px] -translate-x-1/2 -translate-y-[110%] glass px-3 py-3 text-[10px] rounded-xl text-foreground shadow-lg"
              style={{ left: `${hoveredSegmentLeft}%` }}
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: regimeColors[hoveredSegment.regime] || "#666" }} />
                <span className="font-semibold text-[11px]">{hoveredSegment.regime}</span>
              </div>
              <div className="grid grid-cols-[56px_1fr] gap-x-2 gap-y-1 text-muted-foreground">
                <span>Period</span>
                <span>{hoveredSegment.start} – {hoveredSegment.end}</span>
                <span>Duration</span>
                <span>{hoveredSegment.months} month{hoveredSegment.months === 1 ? "" : "s"}</span>
                <span>Share</span>
                <span>{((hoveredSegment.months / totalMonths) * 100).toFixed(1)}% of history</span>
              </div>
              <div className="absolute left-1/2 top-full h-2 w-2 -translate-x-1/2 -translate-y-1/2 rotate-45 border-r border-b border-border bg-[var(--glass-strong-bg)]" />
            </div>
          )}

          <div className="flex h-20 rounded-lg overflow-hidden gap-[1px]">
            {timeline.map((seg, i) => (
              <div
                key={`${seg.regime}-${seg.start}-${seg.end}-${i}`}
                className="cursor-pointer transition-all duration-200"
                style={{
                  width: `${(seg.months / totalMonths) * 100}%`,
                  backgroundColor: regimeColors[seg.regime],
                  opacity: hoveredSeg === i ? 1 : 0.6,
                }}
                onMouseEnter={() => setHoveredSeg(i)}
                onMouseLeave={() => setHoveredSeg(null)}
              />
            ))}
          </div>

          {crisisAnnotations.map((ann) => {
            const yearStart = 2005;
            const pos = ((ann.year - yearStart) / (2026 - yearStart)) * 100;
            return (
              <div key={ann.year} className="absolute" style={{ left: `${pos}%`, top: "84px" }}>
                <div className="w-[1px] h-4 bg-border mx-auto" />
                <span className="text-[8px] text-muted-foreground whitespace-nowrap block -translate-x-1/2">{ann.label}</span>
              </div>
            );
          })}
        </div>
        <div className="flex justify-between mt-10 text-[10px] text-muted-foreground font-mono">
          <span>2005</span><span>2008</span><span>2011</span><span>2014</span><span>2017</span><span>2020</span><span>2023</span><span>2026</span>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass rounded-2xl p-6"
        >
          <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-5">Transition Matrix</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr>
                  <th className="pb-2 text-muted-foreground font-medium"></th>
                  {transitionMatrix.labels.map((l) => (
                    <th key={l} className="pb-2 text-muted-foreground font-medium text-center px-1">{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {transitionMatrix.labels.map((row, ri) => (
                  <tr key={row}>
                    <td className="py-1 pr-2 text-muted-foreground font-medium">{row}</td>
                    {transitionMatrix.data[ri].map((val, ci) => {
                      const intensity = val / 100;
                      return (
                        <td key={ci} className="py-1 text-center px-1">
                          <div
                            className="rounded-md px-1 py-1.5 font-mono text-[10px]"
                            style={{
                              backgroundColor: `hsl(var(--primary) / ${intensity * 0.5})`,
                              color: intensity > 0.3 ? "hsl(var(--primary-foreground))" : "hsl(var(--muted-foreground))",
                            }}
                          >
                            {val.toFixed(1)}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="glass rounded-2xl p-6"
        >
          <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-5">Regime Characteristics</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  {["Regime", "Days", "%", "VIX Mean", "SPX Return", "HY Spread", "Yield Curve"].map((h) => (
                    <th key={h} className="text-left text-[10px] uppercase tracking-wider text-muted-foreground pb-2 font-medium pr-3">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {characteristics.map((r) => (
                  <tr key={`${r.regime}-${r.days}`} className="border-b border-border/50" style={{ borderLeftWidth: 2, borderLeftColor: regimeColors[r.regime] }}>
                    <td className="py-2.5 text-foreground font-medium">{r.regime}</td>
                    <td className="py-2.5 font-mono text-muted-foreground">{r.days.toLocaleString()}</td>
                    <td className="py-2.5 font-mono text-muted-foreground">{r.pct}%</td>
                    <td className="py-2.5 font-mono text-muted-foreground">{r.vixMean.toFixed(2)}</td>
                    <td className={`py-2.5 font-mono ${r.spxReturn >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {r.spxReturn >= 0 ? "+" : ""}{(r.spxReturn * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 font-mono text-muted-foreground">{r.hySpread.toFixed(2)}</td>
                    <td className="py-2.5 font-mono text-muted-foreground">{r.yieldCurve.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Regimes;
