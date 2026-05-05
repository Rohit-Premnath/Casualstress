import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import { useQueries } from "@tanstack/react-query";
import { api } from "@/services/api";
import { regimeColors } from "@/data/mockData";

const parseYearMonthStart = (value: string) => new Date(`${value}-01T00:00:00`);

const parseYearMonthEnd = (value: string) => {
  const [year, month] = value.split("-").map(Number);
  return new Date(year, month, 0);
};

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
  const displayLabels = useMemo(
    () => (transitionMatrix?.labels ?? []).filter((label) => label !== "High Stress"),
    [transitionMatrix],
  );
  const characteristics = useMemo(() => {
    if (!characteristicsQuery.data || !transitionMatrix?.labels?.length) return characteristicsQuery.data;
    const order = displayLabels;
    return [...characteristicsQuery.data].sort(
      (a, b) => order.indexOf(a.regime) - order.indexOf(b.regime),
    ).filter((row) => row.regime !== "High Stress");
  }, [characteristicsQuery.data, transitionMatrix, displayLabels]);

  const totalMonths = timeline?.reduce((a, s) => a + s.months, 0) ?? 0;
  const hoveredSegment = hoveredSeg !== null ? timeline?.[hoveredSeg] : null;

  const hoveredSegmentLeft = useMemo(() => {
    if (hoveredSeg === null || !timeline?.length || totalMonths === 0) return 0;
    const monthsBefore = timeline.slice(0, hoveredSeg).reduce((sum, seg) => sum + seg.months, 0);
    const currentWidth = timeline[hoveredSeg].months;
    return ((monthsBefore + currentWidth / 2) / totalMonths) * 100;
  }, [hoveredSeg, timeline, totalMonths]);

  const historyRange = useMemo(() => {
    if (!timeline?.length) {
      return { label: "Regime History", years: [] as number[] };
    }

    const startDate = parseYearMonthStart(timeline[0].start);
    const endDate = parseYearMonthEnd(timeline[timeline.length - 1].end);
    const startYear = startDate.getFullYear();
    const endYear = endDate.getFullYear();
    const step = Math.max(1, Math.ceil((endYear - startYear) / 6));
    const years: number[] = [];

    for (let year = startYear; year <= endYear; year += step) {
      years.push(year);
    }
    if (years[years.length - 1] !== endYear) {
      years.push(endYear);
    }

    return {
      label: `Regime History ${startYear}\u2013${endYear}`,
      years,
    };
  }, [timeline]);

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
    !displayLabels.length ||
    !transitionMatrix?.data?.length
  ) {
    return (
      <div className="p-6 md:p-8 max-w-[1440px] mx-auto space-y-6">
        <div className="glass rounded-2xl p-8">
          <h2 className="text-lg font-semibold text-foreground mb-2">Live regime data unavailable</h2>
          <p className="text-sm text-muted-foreground">
            This page does not use mock data for its live regime outputs. If you see this state, one or more live regime endpoints failed or returned empty results.
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

        <div className="max-w-lg mx-auto mt-8 rounded-2xl border border-border bg-secondary/30 px-4 py-4 text-left">
          <p className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Live Confidence</p>
          <p className="text-sm text-foreground font-medium">
            The backend currently returns the live confidence for the current regime only.
          </p>
          <p className="text-xs text-muted-foreground mt-2 leading-relaxed">
            This page no longer fabricates a full regime probability distribution. The history, transition matrix, and regime characteristics below are live-backed.
          </p>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="glass rounded-2xl p-6"
      >
        <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-5">{historyRange.label}</h2>
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
        </div>
        <div className="flex justify-between mt-10 text-[10px] text-muted-foreground font-mono">
          {historyRange.years.map((year) => (
            <span key={year}>{year}</span>
          ))}
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
                  {displayLabels.map((label) => (
                    <th key={label} className="pb-2 text-muted-foreground font-medium text-center px-1">{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {displayLabels.map((row) => {
                  const rowIndex = transitionMatrix.labels.indexOf(row);
                  return (
                  <tr key={row}>
                    <td className="py-1 pr-2 text-muted-foreground font-medium">{row}</td>
                    {displayLabels.map((column) => {
                      const colIndex = transitionMatrix.labels.indexOf(column);
                      const val = transitionMatrix.data[rowIndex][colIndex];
                      const intensity = val / 100;
                      return (
                        <td key={colIndex} className="py-1 text-center px-1">
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
                )})}
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
                  {["Regime", "Days", "%", "VIX Mean", "SPX Return", "HY Spread", "Yield Curve"].map((heading) => (
                    <th key={heading} className="text-left text-[10px] uppercase tracking-wider text-muted-foreground pb-2 font-medium pr-3">{heading}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {characteristics.map((row) => (
                  <tr
                    key={`${row.regime}-${row.days}`}
                    className="border-b border-border/50"
                    style={{ borderLeftWidth: 2, borderLeftColor: regimeColors[row.regime] }}
                  >
                    <td className="py-2.5 text-foreground font-medium">{row.regime}</td>
                    <td className="py-2.5 font-mono text-muted-foreground">{row.days.toLocaleString()}</td>
                    <td className="py-2.5 font-mono text-muted-foreground">{row.pct}%</td>
                    <td className="py-2.5 font-mono text-muted-foreground">{row.vixMean.toFixed(2)}</td>
                    <td className={`py-2.5 font-mono ${row.spxReturn >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {row.spxReturn >= 0 ? "+" : ""}
                      {(row.spxReturn * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 font-mono text-muted-foreground">{row.hySpread.toFixed(2)}</td>
                    <td className="py-2.5 font-mono text-muted-foreground">{row.yieldCurve.toFixed(2)}</td>
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
