import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useQueries } from "@tanstack/react-query";
import * as d3 from "d3";
import {
  Search, X, Info, ChevronDown, ChevronRight, ChevronUp, Crosshair,
  ArrowDownLeft, ArrowUpRight, GitCompare, Zap, ArrowRight,
  Eye, Network, Route, Minus, Plus
} from "lucide-react";
import { api, type CausalNode, type CausalEdge, type CausalGraphResponse } from "@/services/api";

// ── Types ──
interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  category: string;
  major?: boolean;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  weight: number;
  lagged?: boolean;
}

// ── Constants ──
const regimeFilters = ['All', 'Calm', 'Normal', 'Elevated', 'Stressed', 'High Stress', 'Crisis'];
const compareOptions = ['None', 'Calm', 'Normal', 'Elevated', 'Stressed', 'High Stress', 'Crisis'];
const graphViews = [
  { id: 'discovery', label: 'Discovery Graph', icon: Network },
  { id: 'strong', label: 'Strong Links', icon: Zap },
  { id: 'transmission', label: 'Transmission Paths', icon: Route },
];

const STRONG_LINK_MIN_WEIGHT = 0.8;
const categoryColors: Record<string, string> = {
  equity: '#3b82f6',
  macro: '#10b981',
  rates: '#f59e0b',
  volatility: '#ef4444',
  commodities: '#eab308',
  'fixed-income': '#a855f7',
  currency: '#06b6d4',
};

// Plain-English category labels for readable path descriptions
const categoryReadable: Record<string, string> = {
  equity: 'Equities', macro: 'Macro', rates: 'Rates', volatility: 'Volatility',
  commodities: 'Commodities', 'fixed-income': 'Bonds', currency: 'FX',
};

/** Discover the strongest 2-step and 3-step directed paths from a set of edges */
function discoverTransmissionPaths(
  edges: CausalEdge[],
  allNodes: CausalNode[],
  focusNode: string | null,
  maxPaths = 8,
) {
  // Build adjacency: source -> [{target, weight}]
  const adj = new Map<string, { target: string; weight: number }[]>();
  edges.forEach(e => {
    if (!adj.has(e.source)) adj.set(e.source, []);
    adj.get(e.source)!.push({ target: e.target, weight: e.weight });
  });

  type Path = { chain: string[]; strength: number };
  const paths: Path[] = [];

  // Enumerate 2-step paths: A -> B -> C
  adj.forEach((neighbours, a) => {
    for (const { target: b, weight: w1 } of neighbours) {
      const bNeighbours = adj.get(b);
      if (!bNeighbours) continue;
      for (const { target: c, weight: w2 } of bNeighbours) {
        if (c === a) continue; // no loops
        paths.push({ chain: [a, b, c], strength: w1 * w2 });
      }
    }
  });

  // Enumerate 3-step paths: A -> B -> C -> D
  adj.forEach((neighbours, a) => {
    for (const { target: b, weight: w1 } of neighbours) {
      const bN = adj.get(b);
      if (!bN) continue;
      for (const { target: c, weight: w2 } of bN) {
        if (c === a) continue;
        const cN = adj.get(c);
        if (!cN) continue;
        for (const { target: d, weight: w3 } of cN) {
          if (d === a || d === b) continue;
          paths.push({ chain: [a, b, c, d], strength: w1 * w2 * w3 });
        }
      }
    }
  });

  // Filter to focus node if selected
  let filtered = focusNode
    ? paths.filter(p => p.chain.includes(focusNode))
    : paths;

  // Sort by combined strength descending
  filtered.sort((a, b) => b.strength - a.strength);

  // Deduplicate: skip paths that are sub-paths of already-selected stronger paths
  const selected: Path[] = [];
  const usedKeys = new Set<string>();
  for (const p of filtered) {
    const key = p.chain.join('->');
    if (usedKeys.has(key)) continue;
    // Check if this path is a subset of an already-selected path
    const isSubset = selected.some(s => {
      const sKey = s.chain.join('->');
      return sKey.includes(key) || key.includes(sKey);
    });
    if (!isSubset) {
      selected.push(p);
      usedKeys.add(key);
    }
    if (selected.length >= maxPaths) break;
  }

  // Build readable labels
  return selected.map(p => {
    const labels = p.chain.map(id => {
      const node = allNodes.find(n => n.id === id);
      return node ? (categoryReadable[node.category] || node.label) : id;
    });
    return {
      chain: p.chain,
      strength: p.strength,
      label: labels.join(' → '),
      interpretation: `${allNodes.find(n => n.id === p.chain[0])?.label || p.chain[0]} transmits through ${p.chain.length - 2} intermediary${p.chain.length > 3 ? ' nodes' : ''} to ${allNodes.find(n => n.id === p.chain[p.chain.length - 1])?.label || p.chain[p.chain.length - 1]}`,
    };
  });
}

// ── Reusable graph renderer ──
function renderGraph(
  svgEl: SVGSVGElement,
  width: number,
  height: number,
  edges: CausalEdge[],
  allNodes: CausalNode[],
  nodePositions: Map<string, { x: number; y: number }>,
  stressDiffHighlight: boolean,
  stressDiffEdgeKeys: Set<string>,
  selectedNode: string | null,
  setSelectedNode: (n: string | null) => void,
  setHoveredNode: (n: string | null) => void,
  setHoveredEdge: (e: { source: string; target: string; weight: number } | null) => void,
) {
  const svg = d3.select(svgEl);
  svg.selectAll('*').remove();

  const nodes: SimNode[] = allNodes.map(n => {
    const pos = nodePositions.get(n.id);
    return { ...n, x: pos ? pos.x : width / 2, y: pos ? pos.y : height / 2, fx: pos ? pos.x : undefined, fy: pos ? pos.y : undefined };
  });

  const links: (SimLink & { compareOnly?: boolean; primaryOnly?: boolean })[] = edges.map(e => ({
    source: e.source,
    target: e.target,
    weight: e.weight,
    lagged: e.weight < 0.6,
  }));

  const defs = svg.append('defs');

  defs.append('marker').attr('id', 'arrowhead').attr('viewBox', '0 -5 10 10')
    .attr('refX', 22).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
    .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', 'hsl(var(--muted-foreground) / 0.3)');

  defs.append('marker').attr('id', 'arrowhead-active').attr('viewBox', '0 -5 10 10')
    .attr('refX', 22).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
    .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', 'hsl(var(--primary))');

  defs.append('marker').attr('id', 'arrowhead-stress').attr('viewBox', '0 -5 10 10')
    .attr('refX', 22).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
    .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', 'hsl(var(--destructive))');

  Object.entries(categoryColors).forEach(([cat, color]) => {
    const filter = defs.append('filter').attr('id', `glow-${cat}`).attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
    filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur');
    filter.append('feFlood').attr('flood-color', color).attr('flood-opacity', '0.25').attr('result', 'color');
    filter.append('feComposite').attr('in', 'color').attr('in2', 'blur').attr('operator', 'in').attr('result', 'shadow');
    const merge = filter.append('feMerge');
    merge.append('feMergeNode').attr('in', 'shadow');
    merge.append('feMergeNode').attr('in', 'SourceGraphic');
  });

  const g = svg.append('g');

  const zoom = d3.zoom<SVGSVGElement, unknown>()
    .scaleExtent([0.3, 3])
    .on('zoom', (event) => g.attr('transform', event.transform));
  svg.call(zoom);

  const link = g.append('g')
    .selectAll<SVGLineElement, typeof links[0]>('line')
    .data(links)
    .join('line')
    .attr('stroke', d => {
      const key = `${(d.source as any).id || d.source}->${(d.target as any).id || d.target}`;
      if (stressDiffHighlight && stressDiffEdgeKeys.has(key)) return 'hsl(var(--destructive) / 0.6)';
      return 'hsl(var(--muted-foreground) / 0.15)';
    })
    .attr('stroke-width', d => Math.max(1, d.weight * 2))
    .attr('stroke-dasharray', d => d.lagged ? '6,4' : 'none')
    .attr('marker-end', d => {
      const key = `${(d.source as any).id || d.source}->${(d.target as any).id || d.target}`;
      if (stressDiffHighlight && stressDiffEdgeKeys.has(key)) return 'url(#arrowhead-stress)';
      return 'url(#arrowhead)';
    })
    .attr('class', 'graph-edge')
    .style('cursor', 'pointer');

  link.on('mouseover', function (event, d) {
    d3.select(this)
      .attr('stroke', 'hsl(var(--primary) / 0.7)')
      .attr('stroke-dasharray', '8,4')
      .attr('marker-end', 'url(#arrowhead-active)');
    (this as SVGLineElement).style.animation = 'dash-flow 0.5s linear infinite';
    const src = typeof d.source === 'string' ? d.source : (d.source as SimNode).id;
    const tgt = typeof d.target === 'string' ? d.target : (d.target as SimNode).id;
    setHoveredEdge({ source: src, target: tgt, weight: d.weight });
  }).on('mouseout', function (_, d) {
    const key = `${(d.source as any).id || d.source}->${(d.target as any).id || d.target}`;
    const isStressDiff = stressDiffHighlight && stressDiffEdgeKeys.has(key);
    d3.select(this)
      .attr('stroke', isStressDiff ? 'hsl(var(--destructive) / 0.6)' : 'hsl(var(--muted-foreground) / 0.15)')
      .attr('stroke-dasharray', (d as SimLink).lagged ? '6,4' : 'none')
      .attr('marker-end', isStressDiff ? 'url(#arrowhead-stress)' : 'url(#arrowhead)');
    (this as SVGLineElement).style.animation = '';
    setHoveredEdge(null);
  });

  const node = g.append('g')
    .selectAll<SVGGElement, SimNode>('g')
    .data(nodes)
    .join('g')
    .attr('cursor', 'pointer');

  node.append('circle')
      .attr('r', d => d.major ? 20 : 14)
      .attr('fill', d => (categoryColors[d.category] || '#666') + '33')
      .attr('stroke', d => categoryColors[d.category] || '#666')
      .attr('stroke-width', 1.5)
      .attr('filter', d => `url(#glow-${d.category})`);

    node.append('text')
      .text(d => d.id.length > 8 ? d.id.slice(0, 7) + '…' : d.id)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', 'hsl(var(--foreground))')
      .attr('font-size', '8px')
    .attr('font-family', 'JetBrains Mono, monospace')
    .attr('pointer-events', 'none');

  node.on('click', (event, d) => {
    event.stopPropagation();
    setSelectedNode(d.id === '__TOGGLE__' ? null : d.id);
  });

  node.on('mouseover', function (_, d) {
    d3.select(this).select('circle')
      .transition().duration(150)
      .attr('r', d.major ? 25 : 18)
      .attr('stroke-width', 2.5);
    setHoveredNode(d.id);
    link.each(function (l: any) {
      const src = typeof l.source === 'string' ? l.source : l.source?.id;
      const tgt = typeof l.target === 'string' ? l.target : l.target?.id;
      if (src === d.id || tgt === d.id) {
        d3.select(this)
          .attr('stroke', 'hsl(var(--primary) / 0.6)')
          .attr('stroke-dasharray', '8,4')
          .attr('marker-end', 'url(#arrowhead-active)');
        (this as SVGLineElement).style.animation = 'dash-flow 0.5s linear infinite';
      }
    });
  }).on('mouseout', function (_, d) {
    d3.select(this).select('circle')
      .transition().duration(150)
      .attr('r', d.major ? 20 : 14)
      .attr('stroke-width', 1.5);
    setHoveredNode(null);
    link.each(function (l: any) {
      const key = `${(l.source as any).id || l.source}->${(l.target as any).id || l.target}`;
      const isStressDiff = stressDiffHighlight && stressDiffEdgeKeys.has(key);
      d3.select(this)
        .attr('stroke', isStressDiff ? 'hsl(var(--destructive) / 0.6)' : 'hsl(var(--muted-foreground) / 0.15)')
        .attr('stroke-dasharray', (l as SimLink).lagged ? '6,4' : '')
        .attr('marker-end', isStressDiff ? 'url(#arrowhead-stress)' : 'url(#arrowhead)');
      (this as SVGLineElement).style.animation = '';
    });
  });

  svg.on('click', () => setSelectedNode(null));

  // Use fixed positions — no simulation force, just place nodes
  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink<SimNode, SimLink>(links).id(d => d.id).distance(120))
    .force('charge', d3.forceManyBody().strength(-400))
    .force('center', d3.forceCenter(width / 2, height / 2).strength(0.05))
    .force('collision', d3.forceCollide(22))
    .alpha(nodePositions.size > 0 ? 0 : 1) // skip simulation if positions are preset
    .on('tick', () => {
      link
        .attr('x1', d => (d.source as SimNode).x!)
        .attr('y1', d => (d.source as SimNode).y!)
        .attr('x2', d => (d.target as SimNode).x!)
        .attr('y2', d => (d.target as SimNode).y!);
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

  // If positions are preset, do a single tick to place everything
  if (nodePositions.size > 0) {
    simulation.tick(1);
    link
      .attr('x1', d => (d.source as SimNode).x!)
      .attr('y1', d => (d.source as SimNode).y!)
      .attr('x2', d => (d.target as SimNode).x!)
      .attr('y2', d => (d.target as SimNode).y!);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  }

  return simulation;
}

const CausalGraph = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const svgLeftRef = useRef<SVGSVGElement>(null);
  const svgRightRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const leftContainerRef = useRef<HTMLDivElement>(null);
  const rightContainerRef = useRef<HTMLDivElement>(null);
  const [graphView, setGraphView] = useState('discovery');
  const [activeRegime, setActiveRegime] = useState('All');
  const [compareRegime, setCompareRegime] = useState('None');
  const [confidence, setConfidence] = useState(0);
  const [search, setSearch] = useState('');
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<{ source: string; target: string; weight: number } | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [leftDims, setLeftDims] = useState({ width: 400, height: 600 });
  const [rightDims, setRightDims] = useState({ width: 400, height: 600 });
  const [stressDiffHighlight, setStressDiffHighlight] = useState(false);
  const [methodNotesOpen, setMethodNotesOpen] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);

  // Shared node positions for synchronized dual panels
  const nodePositionsRef = useRef<Map<string, { x: number; y: number }>>(new Map());

  const isCompareActive = compareRegime !== 'None';

  const regimeQueries = useQueries({
    queries: regimeFilters.map((regime) => ({
      queryKey: ["causal-graph", regime],
      queryFn: () => api.causal.getGraph(regime === "All" ? undefined : regime, 0),
      retry: 1,
      staleTime: 60_000,
    })),
  });

  const graphResponses = useMemo(() => {
    const out: Record<string, CausalGraphResponse | undefined> = {};
    regimeFilters.forEach((regime, index) => {
      out[regime] = regimeQueries[index]?.data;
    });
    return out;
  }, [regimeQueries]);

  const graphNodeMap = useMemo(() => {
    const out = new Map<string, CausalNode>();
    regimeFilters.forEach((regime) => {
      graphResponses[regime]?.nodes?.forEach((node) => {
        if (!out.has(node.id)) out.set(node.id, node);
      });
    });
    return out;
  }, [graphResponses]);

  const causalNodes = useMemo(
    () => Array.from(graphNodeMap.values()),
    [graphNodeMap],
  );

  const causalEdges = useMemo(() => {
    return regimeFilters.flatMap((regime) => {
      const response = graphResponses[regime];
      if (!response?.edges?.length) return [];
      return response.edges.map((edge) => ({
        ...edge,
        regime: regime === "All" ? "ALL" : regime,
      }));
    });
  }, [graphResponses]);

  const graphsLoading = regimeQueries.some((query) => query.isLoading);
  const loadedGraphCount = regimeQueries.filter((query) => !!query.data).length;
  const noGraphsAvailable = loadedGraphCount === 0 && regimeQueries.every((query) => query.isError || !query.data);
  const partialGraphData = loadedGraphCount > 0 && loadedGraphCount < regimeFilters.length;

  // Resize
  useEffect(() => {
    const updateDims = () => {
      if (!isCompareActive && containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
      if (isCompareActive) {
        if (leftContainerRef.current) {
          setLeftDims({ width: leftContainerRef.current.clientWidth, height: leftContainerRef.current.clientHeight });
        }
        if (rightContainerRef.current) {
          setRightDims({ width: rightContainerRef.current.clientWidth, height: rightContainerRef.current.clientHeight });
        }
      }
    };
    updateDims();
    window.addEventListener('resize', updateDims);
    return () => window.removeEventListener('resize', updateDims);
  }, [isCompareActive]);

  // ── Edge filtering ──
  const getEdgesForRegime = useCallback((regime: string) => {
    const baseEdges = causalEdges
      .filter(e => {
        if (regime === 'All') return e.regime === 'ALL';
        return e.regime === regime;
      })
      .filter(e => e.source !== e.target);
    const maxWeight = baseEdges.length > 0 ? Math.max(...baseEdges.map(e => e.weight)) : 0;
    const minWeight = maxWeight > 0 ? maxWeight * (confidence / 100) : 0;
    let edges = baseEdges.filter(e => e.weight >= minWeight);
    if (graphView === 'strong') {
      edges = edges.filter(e => e.weight >= STRONG_LINK_MIN_WEIGHT);
    }
    // For transmission view, don't pre-filter — let path discovery handle it
    return edges;
  }, [confidence, graphView]);

  const filteredEdges = useMemo(() => getEdgesForRegime(activeRegime), [getEdgesForRegime, activeRegime]);

  const compareEdges = useMemo(() => {
    if (compareRegime === 'None') return [];
    return getEdgesForRegime(compareRegime);
  }, [getEdgesForRegime, compareRegime]);

  // ── Computed transmission paths (always computed for lower section, focused in transmission view) ──
  const computedTransmissionPaths = useMemo(() => {
    return discoverTransmissionPaths(filteredEdges, causalNodes, graphView === 'transmission' ? selectedNode : null, graphView === 'transmission' && selectedNode ? 6 : 8);
  }, [filteredEdges, causalNodes, selectedNode, graphView]);

  // Nodes and edges for transmission-path-only rendering
  const transmissionSubgraph = useMemo(() => {
    if (graphView !== 'transmission' || computedTransmissionPaths.length === 0) return null;
    const nodeIds = new Set<string>();
    const edgeKeys = new Set<string>();
    computedTransmissionPaths.forEach(p => {
      p.chain.forEach(id => nodeIds.add(id));
      for (let i = 0; i < p.chain.length - 1; i++) {
        edgeKeys.add(`${p.chain[i]}->${p.chain[i + 1]}`);
      }
    });
    const subEdges = filteredEdges.filter(e => edgeKeys.has(`${e.source}->${e.target}`));
    return { nodeIds, edgeKeys, subEdges };
  }, [graphView, computedTransmissionPaths, filteredEdges]);

  // ── Compare analysis ──
  const compareAnalysis = useMemo(() => {
    if (compareRegime === 'None') return null;
    const primaryKeys = new Set(filteredEdges.map(e => `${e.source}->${e.target}`));
    const compareKeys = new Set(compareEdges.map(e => `${e.source}->${e.target}`));

    const added = compareEdges.filter(e => !primaryKeys.has(`${e.source}->${e.target}`));
    const removed = filteredEdges.filter(e => !compareKeys.has(`${e.source}->${e.target}`));

    const strengthened: { source: string; target: string; delta: number }[] = [];
    const weakened: { source: string; target: string; delta: number }[] = [];

    filteredEdges.forEach(pe => {
      const key = `${pe.source}->${pe.target}`;
      if (compareKeys.has(key)) {
        const ce = compareEdges.find(e => `${e.source}->${e.target}` === key);
        if (ce) {
          const delta = ce.weight - pe.weight;
          if (delta > 0.2) strengthened.push({ source: pe.source, target: pe.target, delta });
          else if (delta < -0.2) weakened.push({ source: pe.source, target: pe.target, delta });
        }
      }
    });

    const primaryAvg = filteredEdges.length > 0 ? filteredEdges.reduce((s, e) => s + e.weight, 0) / filteredEdges.length : 0;
    const compareAvg = compareEdges.length > 0 ? compareEdges.reduce((s, e) => s + e.weight, 0) / compareEdges.length : 0;

    return {
      added: added.sort((a, b) => b.weight - a.weight),
      removed: removed.sort((a, b) => b.weight - a.weight),
      strengthened: strengthened.sort((a, b) => b.delta - a.delta),
      weakened: weakened.sort((a, b) => a.delta - b.delta),
      primaryEdgeCount: filteredEdges.length,
      compareEdgeCount: compareEdges.length,
      avgEdgeWeightShift: compareAvg - primaryAvg,
    };
  }, [filteredEdges, compareEdges, compareRegime]);

  // ── Stress-diff edges ──
  const stressDiffEdgeKeys = useMemo(() => {
    if (!stressDiffHighlight) return new Set<string>();
    const calmEdges = new Set(
      causalEdges.filter(e => e.regime === 'Calm' || e.regime === 'Normal').map(e => `${e.source}->${e.target}`)
    );
    const stressEdges = causalEdges.filter(e => e.regime === 'Stressed' || e.regime === 'Crisis');
    const diffKeys = new Set<string>();
    stressEdges.forEach(e => {
      const key = `${e.source}->${e.target}`;
      if (!calmEdges.has(key)) diffKeys.add(key);
      else {
        const calmWeight = causalEdges.find(ce => (ce.regime === 'Calm' || ce.regime === 'Normal') && ce.source === e.source && ce.target === e.target)?.weight || 0;
        if (e.weight > calmWeight * 1.3) diffKeys.add(key);
      }
    });
    return diffKeys;
  }, [stressDiffHighlight]);

  // ── Most stress-sensitive variables ──
  const stressSensitiveVars = useMemo(() => {
    const calmEdgeMap = new Map<string, number>();
    const stressEdgeMap = new Map<string, number>();
    causalEdges.filter(e => e.regime === 'Calm' || e.regime === 'Normal').forEach(e => {
      calmEdgeMap.set(e.source, (calmEdgeMap.get(e.source) || 0) + 1);
      calmEdgeMap.set(e.target, (calmEdgeMap.get(e.target) || 0) + 1);
    });
    causalEdges.filter(e => e.regime === 'Stressed' || e.regime === 'Crisis').forEach(e => {
      stressEdgeMap.set(e.source, (stressEdgeMap.get(e.source) || 0) + 1);
      stressEdgeMap.set(e.target, (stressEdgeMap.get(e.target) || 0) + 1);
    });
    const allVars = new Set([...calmEdgeMap.keys(), ...stressEdgeMap.keys()]);
    const diffs: { id: string; label: string; calm: number; stress: number; delta: number }[] = [];
    allVars.forEach(v => {
      const calm = calmEdgeMap.get(v) || 0;
      const stress = stressEdgeMap.get(v) || 0;
      const node = causalNodes.find(n => n.id === v);
      diffs.push({ id: v, label: node?.label || v, calm, stress, delta: stress - calm });
    });
    return diffs.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 8);
  }, []);

  // ── Node helpers ──
  const getNodeEdges = useCallback((nodeId: string) => {
    const outgoing = filteredEdges.filter(e => e.source === nodeId);
    const incoming = filteredEdges.filter(e => e.target === nodeId);
    return { outgoing, incoming };
  }, [filteredEdges]);

  const getNodeStats = useCallback((nodeId: string) => {
    const inDeg = filteredEdges.filter(e => e.target === nodeId).length;
    const outDeg = filteredEdges.filter(e => e.source === nodeId).length;
    return { inDegree: inDeg, outDegree: outDeg, degree: inDeg + outDeg };
  }, [filteredEdges]);

  const selectedNodeData = selectedNode ? causalNodes.find(n => n.id === selectedNode) : null;
  const selectedEdges = selectedNode ? getNodeEdges(selectedNode) : null;
  const selectedStats = selectedNode ? getNodeStats(selectedNode) : null;

  // ── Summary stats ──
  const summaryStats = useMemo(() => {
    const nodeIds = new Set<string>();
    filteredEdges.forEach(e => { nodeIds.add(e.source); nodeIds.add(e.target); });
    const avgConf = filteredEdges.length > 0
      ? filteredEdges.reduce((s, e) => s + e.weight, 0) / filteredEdges.length
      : 0;

    const degreeMap: Record<string, number> = {};
    filteredEdges.forEach(e => {
      degreeMap[e.source] = (degreeMap[e.source] || 0) + 1;
      degreeMap[e.target] = (degreeMap[e.target] || 0) + 1;
    });
    const strongestHub = Object.entries(degreeMap).sort((a, b) => b[1] - a[1])[0];

    const catEdges: Record<string, number> = {};
    filteredEdges.forEach(e => {
      const srcNode = causalNodes.find(n => n.id === e.source);
      if (srcNode) catEdges[srcNode.category] = (catEdges[srcNode.category] || 0) + 1;
    });
    const topCluster = Object.entries(catEdges).sort((a, b) => b[1] - a[1])[0];

    const topSensitive = stressSensitiveVars[0];

    return {
      nodes: nodeIds.size,
      edges: filteredEdges.length,
      avgEdgeWeight: avgConf,
      strongestHub: strongestHub ? strongestHub[0] : '—',
      topCluster: topCluster ? topCluster[0] : '—',
      mostStressSensitive: topSensitive?.id || '—',
    };
  }, [filteredEdges, stressSensitiveVars]);

  // ── Regime sensitivity for selected node ──
  const regimeSensitivity = useMemo(() => {
    if (!selectedNode) return null;
    const regimes = ['Calm', 'Normal', 'Elevated', 'Stressed', 'Crisis'];
    let strongestRegime = 'Calm';
    let maxEdges = 0;
    const edgeCounts: Record<string, number> = {};
    regimes.forEach(r => {
      const count = causalEdges.filter(e => e.regime === r && (e.source === selectedNode || e.target === selectedNode)).length;
      edgeCounts[r] = count;
      if (count > maxEdges) { maxEdges = count; strongestRegime = r; }
    });
    const inDiscovery = causalEdges.some(e => e.regime === 'ALL' && (e.source === selectedNode || e.target === selectedNode));
    const inStrongLinks = causalEdges.filter(e => e.regime === 'ALL' && (e.source === selectedNode || e.target === selectedNode) && e.weight >= STRONG_LINK_MIN_WEIGHT).length > 0;
    const calmCount = edgeCounts['Calm'] || 0;
    const stressCount = (edgeCounts['Stressed'] || 0) + (edgeCounts['Crisis'] || 0);
    const edgeBehavior = stressCount > calmCount ? 'gains' : stressCount < calmCount ? 'loses' : 'stable';

    return { strongestRegime, inDiscovery, inStrongLinks, edgeBehavior, edgeCounts };
  }, [selectedNode]);

  // ── Compare node analysis ──
  const selectedNodeCompare = useMemo(() => {
    if (!selectedNode || compareRegime === 'None') return null;
    const primaryIn = filteredEdges.filter(e => e.target === selectedNode).length;
    const primaryOut = filteredEdges.filter(e => e.source === selectedNode).length;
    const compareIn = compareEdges.filter(e => e.target === selectedNode).length;
    const compareOut = compareEdges.filter(e => e.source === selectedNode).length;

    const primaryNeighbors = new Set([
      ...filteredEdges.filter(e => e.source === selectedNode).map(e => e.target),
      ...filteredEdges.filter(e => e.target === selectedNode).map(e => e.source),
    ]);
    const compareNeighbors = new Set([
      ...compareEdges.filter(e => e.source === selectedNode).map(e => e.target),
      ...compareEdges.filter(e => e.target === selectedNode).map(e => e.source),
    ]);
    const newNeighbors = [...compareNeighbors].filter(n => !primaryNeighbors.has(n));
    const lostNeighbors = [...primaryNeighbors].filter(n => !compareNeighbors.has(n));

    return {
      primaryEdges: primaryIn + primaryOut,
      compareEdges: compareIn + compareOut,
      newNeighbors,
      lostNeighbors,
      pairEdgeBehavior:
        (compareIn + compareOut) > (primaryIn + primaryOut)
          ? 'gains'
          : (compareIn + compareOut) < (primaryIn + primaryOut)
            ? 'loses'
            : 'stable',
    };
  }, [selectedNode, filteredEdges, compareEdges, compareRegime]);

  // ── Search ──
  const searchMatches = useMemo(() => {
    if (!search) return new Set<string>();
    const q = search.toLowerCase();
    return new Set(
      causalNodes
        .filter(n => n.id.toLowerCase().includes(q) || n.label.toLowerCase().includes(q))
        .map(n => n.id)
    );
  }, [search]);

  // ── SINGLE graph rendering (non-compare mode) ──
  useEffect(() => {
    if (isCompareActive) return;
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const { width, height } = dimensions;
    const isTransmission = graphView === 'transmission' && transmissionSubgraph;
    const activeEdges = isTransmission ? transmissionSubgraph.subEdges : filteredEdges;
    const activeNodeIds = isTransmission
      ? transmissionSubgraph.nodeIds
      : new Set(activeEdges.flatMap(e => [e.source, e.target]));

    const nodes: SimNode[] = causalNodes
      .filter(n => activeNodeIds.has(n.id))
      .map(n => ({ ...n }));

    const links: (SimLink & { compareOnly?: boolean; primaryOnly?: boolean })[] =
      activeEdges.map(e => ({
        source: e.source,
        target: e.target,
        weight: e.weight,
        lagged: e.weight < 0.6,
      }));

    const defs = svg.append('defs');

    defs.append('marker').attr('id', 'arrowhead').attr('viewBox', '0 -5 10 10')
      .attr('refX', 22).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
      .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', 'hsl(var(--muted-foreground) / 0.3)');

    defs.append('marker').attr('id', 'arrowhead-active').attr('viewBox', '0 -5 10 10')
      .attr('refX', 22).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
      .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', 'hsl(var(--primary))');

    defs.append('marker').attr('id', 'arrowhead-stress').attr('viewBox', '0 -5 10 10')
      .attr('refX', 22).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
      .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', 'hsl(var(--destructive))');

    Object.entries(categoryColors).forEach(([cat, color]) => {
      const filter = defs.append('filter').attr('id', `glow-${cat}`).attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
      filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur');
      filter.append('feFlood').attr('flood-color', color).attr('flood-opacity', '0.25').attr('result', 'color');
      filter.append('feComposite').attr('in', 'color').attr('in2', 'blur').attr('operator', 'in').attr('result', 'shadow');
      const merge = filter.append('feMerge');
      merge.append('feMergeNode').attr('in', 'shadow');
      merge.append('feMergeNode').attr('in', 'SourceGraphic');
    });

    const g = svg.append('g');

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => g.attr('transform', event.transform));
    svg.call(zoom);

    const link = g.append('g')
      .selectAll<SVGLineElement, typeof links[0]>('line')
      .data(links)
      .join('line')
      .attr('stroke', d => {
        const key = `${(d.source as any).id || d.source}->${(d.target as any).id || d.target}`;
        if (stressDiffHighlight && stressDiffEdgeKeys.has(key)) return 'hsl(var(--destructive) / 0.6)';
        if (isTransmission) return 'hsl(var(--primary) / 0.5)';
        return 'hsl(var(--muted-foreground) / 0.15)';
      })
      .attr('stroke-width', d => isTransmission ? Math.max(2, d.weight * 3.5) : Math.max(1, d.weight * 2))
      .attr('stroke-dasharray', d => d.lagged ? '6,4' : 'none')
      .attr('marker-end', d => {
        const key = `${(d.source as any).id || d.source}->${(d.target as any).id || d.target}`;
        if (stressDiffHighlight && stressDiffEdgeKeys.has(key)) return 'url(#arrowhead-stress)';
        return 'url(#arrowhead)';
      })
      .attr('class', 'graph-edge')
      .style('cursor', 'pointer');

    link.on('mouseover', function (event, d) {
      d3.select(this)
        .attr('stroke', 'hsl(var(--primary) / 0.7)')
        .attr('stroke-dasharray', '8,4')
        .attr('marker-end', 'url(#arrowhead-active)');
      (this as SVGLineElement).style.animation = 'dash-flow 0.5s linear infinite';
      const src = typeof d.source === 'string' ? d.source : (d.source as SimNode).id;
      const tgt = typeof d.target === 'string' ? d.target : (d.target as SimNode).id;
      setHoveredEdge({ source: src, target: tgt, weight: d.weight });
    }).on('mouseout', function (_, d) {
      const key = `${(d.source as any).id || d.source}->${(d.target as any).id || d.target}`;
      const isStressDiff = stressDiffHighlight && stressDiffEdgeKeys.has(key);
      d3.select(this)
        .attr('stroke', isStressDiff ? 'hsl(var(--destructive) / 0.6)' : 'hsl(var(--muted-foreground) / 0.15)')
        .attr('stroke-dasharray', (d as SimLink).lagged ? '6,4' : 'none')
        .attr('marker-end', isStressDiff ? 'url(#arrowhead-stress)' : 'url(#arrowhead)');
      (this as SVGLineElement).style.animation = '';
      setHoveredEdge(null);
    });

    const node = g.append('g')
      .selectAll<SVGGElement, SimNode>('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(d3.drag<SVGGElement, SimNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null; d.fy = null;
        }));

    node.append('circle')
      .attr('r', d => d.major ? 20 : 14)
      .attr('fill', d => (categoryColors[d.category] || '#666') + '33')
      .attr('stroke', d => categoryColors[d.category] || '#666')
      .attr('stroke-width', 1.5)
      .attr('filter', d => `url(#glow-${d.category})`);

    node.append('text')
      .text(d => d.id.length > 8 ? d.id.slice(0, 7) + '…' : d.id)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', 'hsl(var(--foreground))')
      .attr('font-size', '8px')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('pointer-events', 'none');

    node.on('click', (event, d) => {
      event.stopPropagation();
      setSelectedNode(prev => prev === d.id ? null : d.id);
    });

    node.on('mouseover', function (_, d) {
      d3.select(this).select('circle')
        .transition().duration(150)
        .attr('r', d.major ? 25 : 18)
        .attr('stroke-width', 2.5);
      setHoveredNode(d.id);
      link.each(function (l: any) {
        const src = typeof l.source === 'string' ? l.source : l.source?.id;
        const tgt = typeof l.target === 'string' ? l.target : l.target?.id;
        if (src === d.id || tgt === d.id) {
          d3.select(this)
            .attr('stroke', 'hsl(var(--primary) / 0.6)')
            .attr('stroke-dasharray', '8,4')
            .attr('marker-end', 'url(#arrowhead-active)');
          (this as SVGLineElement).style.animation = 'dash-flow 0.5s linear infinite';
        }
      });
    }).on('mouseout', function (_, d) {
      d3.select(this).select('circle')
        .transition().duration(150)
        .attr('r', d.major ? 20 : 14)
        .attr('stroke-width', 1.5);
      setHoveredNode(null);
      if (!selectedNode) {
        link.each(function (l: any) {
          const key = `${(l.source as any).id || l.source}->${(l.target as any).id || l.target}`;
          const isStressDiff = stressDiffHighlight && stressDiffEdgeKeys.has(key);
          d3.select(this)
            .attr('stroke', isStressDiff ? 'hsl(var(--destructive) / 0.6)' : 'hsl(var(--muted-foreground) / 0.15)')
            .attr('stroke-dasharray', (l as SimLink).lagged ? '6,4' : '')
            .attr('marker-end', isStressDiff ? 'url(#arrowhead-stress)' : 'url(#arrowhead)');
          (this as SVGLineElement).style.animation = '';
        });
      }
    });

    svg.on('click', () => setSelectedNode(null));

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink<SimNode, SimLink>(links).id(d => d.id).distance(isTransmission ? 180 : 120))
      .force('charge', d3.forceManyBody().strength(isTransmission ? -800 : -400))
      .force('center', d3.forceCenter(width / 2, height / 2).strength(0.05))
      .force('collision', d3.forceCollide(isTransmission ? 40 : 22))
      .on('tick', () => {
        link
          .attr('x1', d => (d.source as SimNode).x!)
          .attr('y1', d => (d.source as SimNode).y!)
          .attr('x2', d => (d.target as SimNode).x!)
          .attr('y2', d => (d.target as SimNode).y!);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
      });

    // Store settled positions for compare mode
    simulation.on('end', () => {
      const posMap = new Map<string, { x: number; y: number }>();
      nodes.forEach(n => {
        if (n.x != null && n.y != null) posMap.set(n.id, { x: n.x, y: n.y });
      });
      nodePositionsRef.current = posMap;
    });

    return () => { simulation.stop(); };
  }, [dimensions, filteredEdges, stressDiffHighlight, stressDiffEdgeKeys, isCompareActive, graphView, transmissionSubgraph]);

  // ── DUAL graph rendering (compare mode) ──
  useEffect(() => {
    if (!isCompareActive) return;
    if (!svgLeftRef.current || !svgRightRef.current) return;

    // First compute shared positions if we don't have them yet
    // We use ALL edges (union of both regimes) to compute a stable layout
    const allEdgeSet = new Map<string, { source: string; target: string; weight: number }>();
    filteredEdges.forEach(e => allEdgeSet.set(`${e.source}->${e.target}`, e));
    compareEdges.forEach(e => {
      const key = `${e.source}->${e.target}`;
      if (!allEdgeSet.has(key)) allEdgeSet.set(key, e);
    });
    const unionEdges = [...allEdgeSet.values()];
    const unionNodeIds = new Set<string>();
    unionEdges.forEach(e => {
      unionNodeIds.add(e.source);
      unionNodeIds.add(e.target);
    });

    const w = leftDims.width;
    const h = leftDims.height;

    // Run a headless simulation to get positions
    const simNodes: SimNode[] = causalNodes.filter(n => unionNodeIds.has(n.id)).map(n => {
      const existing = nodePositionsRef.current.get(n.id);
      // Scale existing positions to new panel width
      if (existing) {
        return { ...n, x: existing.x * (w / dimensions.width), y: existing.y * (h / dimensions.height) };
      }
      return { ...n };
    });
    const simLinks: SimLink[] = unionEdges.map(e => ({
      source: e.source,
      target: e.target,
      weight: e.weight,
      lagged: e.weight < 0.6,
    }));

    const layoutSim = d3.forceSimulation(simNodes)
      .force('link', d3.forceLink<SimNode, SimLink>(simLinks).id(d => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(w / 2, h / 2).strength(0.05))
      .force('collision', d3.forceCollide(22))
      .stop();

    // Run 300 ticks to settle
    for (let i = 0; i < 300; i++) layoutSim.tick();

    const sharedPositions = new Map<string, { x: number; y: number }>();
    simNodes.forEach(n => {
      if (n.x != null && n.y != null) sharedPositions.set(n.id, { x: n.x, y: n.y });
    });

    const handleSelectNode = (id: string | null) => {
      setSelectedNode(prev => id === null ? null : prev === id ? null : id);
    };

    // Render left panel (primary regime)
    const simLeft = renderGraph(
      svgLeftRef.current, w, h, filteredEdges, causalNodes.filter(n => unionNodeIds.has(n.id)), sharedPositions,
      stressDiffHighlight, stressDiffEdgeKeys,
      null, handleSelectNode, setHoveredNode, setHoveredEdge
    );

    // Render right panel (compare regime)
    const simRight = renderGraph(
      svgRightRef.current, rightDims.width, rightDims.height, compareEdges, causalNodes.filter(n => unionNodeIds.has(n.id)), sharedPositions,
      stressDiffHighlight, stressDiffEdgeKeys,
      null, handleSelectNode, setHoveredNode, setHoveredEdge
    );

    return () => {
      simLeft.stop();
      simRight.stop();
    };
  }, [isCompareActive, leftDims, rightDims, filteredEdges, compareEdges, stressDiffHighlight, stressDiffEdgeKeys, dimensions]);

  // ── Selection highlight effect (single mode only) ──
  useEffect(() => {
    if (isCompareActive) return;
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    if (selectedNode) {
      const connectedIds = new Set<string>();
      connectedIds.add(selectedNode);
      filteredEdges.forEach(e => {
        if (e.source === selectedNode) connectedIds.add(e.target);
        if (e.target === selectedNode) connectedIds.add(e.source);
      });
      svg.selectAll<SVGGElement, SimNode>('g g g').filter(function () {
        return this.querySelector('circle') !== null;
      }).each(function (d: any) {
        d3.select(this).transition().duration(200).style('opacity', connectedIds.has(d?.id) ? 1 : 0.15);
      });
      svg.selectAll<SVGLineElement, SimLink>('line').each(function (d: any) {
        const src = typeof d.source === 'string' ? d.source : d.source?.id;
        const tgt = typeof d.target === 'string' ? d.target : d.target?.id;
        const isActive = src === selectedNode || tgt === selectedNode;
        d3.select(this)
          .transition().duration(200)
          .attr('stroke', isActive ? 'hsl(var(--primary) / 0.6)' : 'hsl(var(--muted-foreground) / 0.05)')
          .attr('marker-end', isActive ? 'url(#arrowhead-active)' : 'url(#arrowhead)');
        if (isActive) {
          (this as SVGLineElement).style.strokeDasharray = '8,4';
          (this as SVGLineElement).style.animation = 'dash-flow 0.5s linear infinite';
        } else {
          (this as SVGLineElement).style.strokeDasharray = '';
          (this as SVGLineElement).style.animation = '';
        }
      });
    } else {
      svg.selectAll('g g g').transition().duration(200).style('opacity', 1);
      svg.selectAll('line').each(function (l: any) {
        const key = `${(l.source as any).id || l.source}->${(l.target as any).id || l.target}`;
        const isStressDiff = stressDiffHighlight && stressDiffEdgeKeys.has(key);
        d3.select(this).transition().duration(200)
          .attr('stroke', isStressDiff ? 'hsl(var(--destructive) / 0.6)' : 'hsl(var(--muted-foreground) / 0.15)')
          .attr('marker-end', isStressDiff ? 'url(#arrowhead-stress)' : 'url(#arrowhead)');
        (this as SVGLineElement).style.strokeDasharray = (l as SimLink).lagged ? '6,4' : '';
        (this as SVGLineElement).style.animation = '';
      });
    }
  }, [selectedNode, filteredEdges, stressDiffHighlight, stressDiffEdgeKeys, isCompareActive]);

  // ── Selection highlight effect (compare mode) ──
  useEffect(() => {
    if (!isCompareActive) return;
    const svgEls = [svgLeftRef.current, svgRightRef.current];
    const edgeSets = [filteredEdges, compareEdges];

    svgEls.forEach((el, idx) => {
      if (!el) return;
      const svg = d3.select(el);
      const edges = edgeSets[idx];

      if (selectedNode) {
        const connectedIds = new Set<string>();
        connectedIds.add(selectedNode);
        edges.forEach(e => {
          if (e.source === selectedNode) connectedIds.add(e.target);
          if (e.target === selectedNode) connectedIds.add(e.source);
        });

        svg.selectAll<SVGGElement, SimNode>('g g g').filter(function () {
          return this.querySelector('circle') !== null;
        }).each(function (d: any) {
          d3.select(this).transition().duration(200).style('opacity', connectedIds.has(d?.id) ? 1 : 0.15);
        });

        svg.selectAll<SVGLineElement, SimLink>('line').each(function (d: any) {
          const src = typeof d.source === 'string' ? d.source : d.source?.id;
          const tgt = typeof d.target === 'string' ? d.target : d.target?.id;
          const isActive = src === selectedNode || tgt === selectedNode;
          d3.select(this)
            .transition().duration(200)
            .attr('stroke', isActive ? 'hsl(var(--primary) / 0.6)' : 'hsl(var(--muted-foreground) / 0.05)')
            .attr('marker-end', isActive ? 'url(#arrowhead-active)' : 'url(#arrowhead)');
          if (isActive) {
            (this as SVGLineElement).style.strokeDasharray = '8,4';
            (this as SVGLineElement).style.animation = 'dash-flow 0.5s linear infinite';
          } else {
            (this as SVGLineElement).style.strokeDasharray = '';
            (this as SVGLineElement).style.animation = '';
          }
        });
      } else {
        svg.selectAll('g g g').transition().duration(200).style('opacity', 1);
        svg.selectAll('line').each(function (l: any) {
          const key = `${(l.source as any).id || l.source}->${(l.target as any).id || l.target}`;
          const isStressDiff = stressDiffHighlight && stressDiffEdgeKeys.has(key);
          d3.select(this).transition().duration(200)
            .attr('stroke', isStressDiff ? 'hsl(var(--destructive) / 0.6)' : 'hsl(var(--muted-foreground) / 0.15)')
            .attr('marker-end', isStressDiff ? 'url(#arrowhead-stress)' : 'url(#arrowhead)');
          (this as SVGLineElement).style.strokeDasharray = (l as SimLink).lagged ? '6,4' : '';
          (this as SVGLineElement).style.animation = '';
        });
      }
    });
  }, [selectedNode, filteredEdges, compareEdges, stressDiffHighlight, stressDiffEdgeKeys, isCompareActive]);

  // ── Search highlight effect ──
  useEffect(() => {
    if (isCompareActive) return;
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    if (searchMatches.size > 0) {
      svg.selectAll<SVGGElement, SimNode>('g g g').filter(function () {
        return this.querySelector('circle') !== null;
      }).each(function (d: any) {
        d3.select(this).transition().duration(200).style('opacity', searchMatches.has(d?.id) ? 1 : 0.15);
      });
    } else if (!selectedNode) {
      svg.selectAll('g g g').transition().duration(200).style('opacity', 1);
    }
  }, [searchMatches, selectedNode, isCompareActive]);

  const hoveredNodeData = hoveredNode ? causalNodes.find(n => n.id === hoveredNode) : null;
  const hoveredStats = hoveredNode ? getNodeStats(hoveredNode) : null;

  if (graphsLoading && loadedGraphCount === 0) {
    return (
      <div className="p-6 md:p-8 max-w-[1440px] mx-auto">
        <div className="glass rounded-2xl p-8 text-sm text-muted-foreground">Loading live causal graph data...</div>
      </div>
    );
  }

  if (noGraphsAvailable || !graphResponses.All) {
    return (
      <div className="p-6 md:p-8 max-w-[1440px] mx-auto">
        <div className="glass rounded-2xl p-8">
          <h2 className="text-lg font-semibold text-foreground mb-2">Live causal graph unavailable</h2>
          <p className="text-sm text-muted-foreground">
            This page no longer falls back to mock graph data. If you see this state, the live causal graph endpoints failed or returned empty results.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-3.5rem)] flex flex-col">
      {partialGraphData && (
        <div className="bg-amber-500/10 border-b border-amber-500/20 px-4 py-2 text-[11px] text-amber-500 shrink-0">
          Partial live mode: some regime graph queries were unavailable, but this page is using the live graphs that did load.
        </div>
      )}
      {/* ── Explanatory strip ── */}
      <div className="bg-secondary/50 border-b border-border px-4 py-1 flex items-center gap-2 shrink-0">
        <Info className="w-3 h-3 text-muted-foreground shrink-0" />
        <p className="text-[10px] text-muted-foreground leading-snug">
          This explorer shows learned market relationship structure across regimes. <span className="text-foreground/70">Some views are direct graph outputs</span>, while comparison overlays summarize how the network changes under stress.
        </p>
      </div>

      {/* ── Top Control Area ── */}
      <div className="glass border-b border-border relative z-20 px-4 py-2 shrink-0">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
          <h1 className="text-sm font-medium text-foreground whitespace-nowrap mr-1">Causal Graph Explorer</h1>

          {/* Regime */}
          <div className="flex flex-col gap-0.5">
            <span className="text-[9px] text-muted-foreground uppercase tracking-widest">Regime</span>
            <div className="flex items-center gap-0.5 bg-secondary rounded-lg p-0.5">
              {regimeFilters.map(r => (
                <button key={r} onClick={() => setActiveRegime(r)}
                  className={`px-2 py-1 text-[10px] rounded-md transition-all font-medium ${
                    activeRegime === r
                      ? 'bg-primary/15 text-primary border border-primary/25'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}>{r}</button>
              ))}
            </div>
          </div>

          {/* Compare To */}
          <div className="flex flex-col gap-0.5">
            <span className="text-[9px] text-muted-foreground uppercase tracking-widest">Compare To</span>
            <div className="flex items-center gap-0.5 bg-secondary rounded-lg p-0.5">
              {compareOptions.map(r => (
                <button key={r} onClick={() => setCompareRegime(r)}
                  className={`px-2 py-1 text-[10px] rounded-md transition-all font-medium ${
                    compareRegime === r
                      ? r === 'None'
                        ? 'bg-muted text-foreground border border-border'
                        : 'bg-accent/15 text-accent border border-accent/25'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}>{r}</button>
              ))}
            </div>
          </div>

          {/* Min Relative Edge Weight */}
          <div className="flex flex-col gap-0.5">
            <span className="text-[9px] text-muted-foreground uppercase tracking-widest">Min Relative Edge Weight</span>
            <div className="flex items-center gap-1.5">
              <input type="range" min="0" max="100" value={confidence} onChange={e => setConfidence(+e.target.value)}
                className="w-16 accent-primary" />
              <span className="font-mono text-[10px] text-foreground w-7">{confidence}%</span>
            </div>
          </div>

          {/* Search */}
          <div className="flex flex-col gap-0.5">
            <span className="text-[9px] text-muted-foreground uppercase tracking-widest">Search Variable</span>
            <div className="relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-muted-foreground" />
              <input value={search} onChange={e => setSearch(e.target.value)}
                placeholder="Ticker or label..."
                className="bg-secondary rounded-md pl-7 pr-2 py-1 text-[10px] text-foreground placeholder-muted-foreground border-none outline-none focus:ring-1 focus:ring-primary/30 w-32" />
            </div>
          </div>

          {/* Graph View */}
          <div className="flex flex-col gap-0.5 ml-auto">
            <span className="text-[9px] text-muted-foreground uppercase tracking-widest">Graph View</span>
            <div className="flex items-center gap-0.5 bg-secondary rounded-lg p-0.5">
              {graphViews.map(v => (
                <button key={v.id} onClick={() => setGraphView(v.id)}
                  className={`flex items-center gap-1 px-2 py-1 text-[10px] rounded-md transition-all font-medium ${
                    graphView === v.id
                      ? 'bg-primary/15 text-primary border border-primary/25'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}>
                  <v.icon className="w-3 h-3" />
                  {v.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Summary Cards ── */}
      <div className="glass border-b border-border px-4 py-1.5 flex items-center gap-2 shrink-0 overflow-x-auto">
        {[
          { label: 'Nodes', value: summaryStats.nodes },
          { label: 'Edges', value: summaryStats.edges },
          { label: 'Avg Edge Weight', value: summaryStats.avgEdgeWeight.toFixed(2) },
          { label: 'Strongest Hub', value: summaryStats.strongestHub },
          { label: 'Top Cluster', value: summaryStats.topCluster.replace('-', ' ') },
          { label: 'Stress-Sensitive', value: summaryStats.mostStressSensitive },
        ].map(s => (
          <div key={s.label} className="bg-secondary/60 rounded-md px-2 py-1 flex items-center gap-1.5 whitespace-nowrap">
            <span className="text-[9px] text-muted-foreground uppercase tracking-wider">{s.label}</span>
            <span className="font-mono text-[10px] text-foreground font-medium capitalize">{s.value}</span>
          </div>
        ))}

        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setStressDiffHighlight(!stressDiffHighlight)}
            className={`flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium transition-all ${
              stressDiffHighlight
                ? 'bg-destructive/15 text-destructive border border-destructive/25'
                : 'bg-secondary/60 text-muted-foreground hover:text-foreground'
            }`}
          >
            <Zap className="w-3 h-3" />
            Stress Changes
          </button>
        </div>
      </div>

      {/* ── Main Content ── */}
      <div className="flex-1 flex relative overflow-hidden min-h-0">
        {/* Graph Canvas Area */}
        {!isCompareActive ? (
          /* Single graph mode */
          <div ref={containerRef} className="flex-1 relative">
            <svg ref={svgRef} width={dimensions.width} height={dimensions.height} />

            {/* Transmission mode overlay label */}
            {graphView === 'transmission' && (
              <div className="absolute top-4 right-4 z-10 glass-strong rounded-lg px-3 py-2">
                <div className="flex items-center gap-2">
                  <Route className="w-3.5 h-3.5 text-primary" />
                  <span className="text-[11px] font-medium text-foreground">Transmission Paths</span>
                </div>
                <p className="text-[10px] text-muted-foreground mt-1">
                  {selectedNode
                    ? `Showing paths through ${selectedNode}`
                    : `Top ${computedTransmissionPaths.length} directed chains`}
                </p>
              </div>
            )}

            {/* Hover tooltip (node) */}
            <AnimatePresence>
              {hoveredNode && hoveredNodeData && !selectedNode && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 4 }}
                  className="absolute top-4 left-4 glass-strong rounded-xl p-4 z-20 min-w-[200px]"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: categoryColors[hoveredNodeData.category] }} />
                    <span className="font-mono text-sm font-semibold text-foreground">{hoveredNodeData.id}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mb-2">{hoveredNodeData.label}</p>
                  <div className="flex gap-4 text-[10px]">
                    <span className="text-muted-foreground">In: <span className="text-foreground font-mono">{hoveredStats?.inDegree}</span></span>
                    <span className="text-muted-foreground">Out: <span className="text-foreground font-mono">{hoveredStats?.outDegree}</span></span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Hover tooltip (edge) */}
            <AnimatePresence>
              {hoveredEdge && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 4 }}
                  className="absolute bottom-4 left-4 glass-strong rounded-xl p-3 z-20"
                >
                  <div className="flex items-center gap-2 text-xs">
                    <span className="font-mono text-foreground">{hoveredEdge.source}</span>
                    <ArrowRight className="w-3 h-3 text-primary" />
                    <span className="font-mono text-foreground">{hoveredEdge.target}</span>
                    <span className="text-muted-foreground ml-2">w: {hoveredEdge.weight.toFixed(2)}</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ) : (
          /* Dual panel compare mode */
          <div className="flex-1 flex flex-col relative">
            {/* Compare mode header strip */}
            <div className="glass-strong border-b border-border px-4 py-2 flex items-center justify-center gap-4 shrink-0 z-10">
              <span className="text-[11px] text-muted-foreground">Comparing</span>
              <span className="text-xs text-primary font-semibold">{activeRegime}</span>
              <ArrowRight className="w-3.5 h-3.5 text-muted-foreground" />
              <span className="text-xs text-accent font-semibold">{compareRegime}</span>
              <div className="w-px h-3.5 bg-border mx-2" />
              <span className="text-[10px] text-muted-foreground">Synchronized node positions · Same variable = same location</span>
            </div>

            {/* Two side-by-side panels */}
            <div className="flex-1 flex min-h-0">
              {/* Left panel */}
              <div ref={leftContainerRef} className="flex-1 relative border-r border-border/50">
                <div className="absolute top-3 left-3 z-10 glass-strong rounded-lg px-3 py-1.5 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-primary" />
                  <span className="text-[11px] font-semibold text-primary">{activeRegime}</span>
                  <span className="text-[10px] text-muted-foreground ml-1">{filteredEdges.length} edges</span>
                </div>
                <svg ref={svgLeftRef} width={leftDims.width} height={leftDims.height} />
              </div>

              {/* Right panel */}
              <div ref={rightContainerRef} className="flex-1 relative">
                <div className="absolute top-3 left-3 z-10 glass-strong rounded-lg px-3 py-1.5 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-accent" />
                  <span className="text-[11px] font-semibold text-accent">{compareRegime}</span>
                  <span className="text-[10px] text-muted-foreground ml-1">{compareEdges.length} edges</span>
                </div>
                <svg ref={svgRightRef} width={rightDims.width} height={rightDims.height} />
              </div>
            </div>

            {/* Hover tooltips for compare mode */}
            <AnimatePresence>
              {hoveredEdge && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 4 }}
                  className="absolute bottom-4 left-1/2 -translate-x-1/2 glass-strong rounded-xl p-3 z-20"
                >
                  <div className="flex items-center gap-2 text-xs">
                    <span className="font-mono text-foreground">{hoveredEdge.source}</span>
                    <ArrowRight className="w-3 h-3 text-primary" />
                    <span className="font-mono text-foreground">{hoveredEdge.target}</span>
                    <span className="text-muted-foreground ml-2">w: {hoveredEdge.weight.toFixed(2)}</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}

        {/* ── Right Side Panel ── */}
        <AnimatePresence mode="wait">
          {selectedNode && selectedNodeData ? (
            <motion.div
              key="detail"
              initial={{ x: 280 }} animate={{ x: 0 }} exit={{ x: 280 }}
              transition={{ duration: 0.2 }}
              className="w-[280px] glass-strong border-l border-border overflow-y-auto shrink-0"
            >
              <div className="p-3">
                {/* Header */}
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-foreground text-sm">{selectedNodeData.id}</h3>
                  <button onClick={() => setSelectedNode(null)} className="text-muted-foreground hover:text-foreground">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Node Details */}
                <div className="mb-3">
                  <span className="inline-block px-2 py-0.5 rounded-full text-[9px] font-medium mb-1.5"
                    style={{ backgroundColor: categoryColors[selectedNodeData.category] + '22', color: categoryColors[selectedNodeData.category] }}>
                    {selectedNodeData.category}
                  </span>
                  <p className="text-[11px] text-muted-foreground mb-2">{selectedNodeData.label}</p>

                  <div className="grid grid-cols-3 gap-1.5 mb-3">
                    {[
                      { label: 'In', value: selectedStats?.inDegree },
                      { label: 'Out', value: selectedStats?.outDegree },
                      { label: 'Total', value: selectedStats?.degree },
                    ].map(s => (
                      <div key={s.label} className="bg-secondary/60 rounded-md px-1.5 py-1.5 text-center">
                        <div className="font-mono text-foreground text-xs">{s.value}</div>
                        <div className="text-[8px] text-muted-foreground">{s.label}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Strongest Parents */}
                {selectedEdges && (
                  <>
                    <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2 flex items-center gap-1.5">
                      <ArrowDownLeft className="w-3 h-3" />
                      Strongest Parents ({selectedEdges.incoming.length})
                    </h4>
                    {selectedEdges.incoming.length === 0 && (
                      <p className="text-[11px] text-muted-foreground/60 mb-3 italic">No incoming edges</p>
                    )}
                    {selectedEdges.incoming
                      .sort((a, b) => b.weight - a.weight)
                      .slice(0, 6)
                      .map((e, i) => (
                      <div key={i} className="flex justify-between text-xs py-1.5 border-b border-border/30 hover:bg-secondary/50 cursor-pointer transition-colors">
                        <div className="flex items-center gap-1.5">
                          <span className="text-muted-foreground text-[10px]">→</span>
                          <span className="text-foreground font-mono">{e.source}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-10 h-1 rounded-full bg-muted overflow-hidden">
                            <div className="h-full rounded-full bg-primary" style={{ width: `${(e.weight / 2.5) * 100}%` }} />
                          </div>
                          <span className="text-muted-foreground font-mono w-8">{e.weight.toFixed(2)}</span>
                        </div>
                      </div>
                    ))}

                    <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mt-4 mb-2 flex items-center gap-1.5">
                      <ArrowUpRight className="w-3 h-3" />
                      Strongest Children ({selectedEdges.outgoing.length})
                    </h4>
                    {selectedEdges.outgoing.length === 0 && (
                      <p className="text-[11px] text-muted-foreground/60 mb-3 italic">No outgoing edges</p>
                    )}
                    {selectedEdges.outgoing
                      .sort((a, b) => b.weight - a.weight)
                      .slice(0, 6)
                      .map((e, i) => (
                      <div key={i} className="flex justify-between text-xs py-1.5 border-b border-border/30 hover:bg-secondary/50 cursor-pointer transition-colors">
                        <div className="flex items-center gap-1.5">
                          <span className="text-muted-foreground text-[10px]">→</span>
                          <span className="text-foreground font-mono">{e.target}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-10 h-1 rounded-full bg-muted overflow-hidden">
                            <div className="h-full rounded-full bg-primary" style={{ width: `${(e.weight / 2.5) * 100}%` }} />
                          </div>
                          <span className="text-muted-foreground font-mono w-8">{e.weight.toFixed(2)}</span>
                        </div>
                      </div>
                    ))}
                  </>
                )}

                {/* Regime Sensitivity */}
                {regimeSensitivity && (
                  <div className="mt-3">
                    <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2">Regime Sensitivity</h4>
                    <div className="bg-secondary/60 rounded-lg p-3 space-y-2 text-[11px]">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Strongest in</span>
                        <span className="text-foreground font-medium">{regimeSensitivity.strongestRegime}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Discovery graph</span>
                        <span className={regimeSensitivity.inDiscovery ? 'text-accent' : 'text-muted-foreground'}>
                          {regimeSensitivity.inDiscovery ? 'Yes' : 'No'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Strong links</span>
                        <span className={regimeSensitivity.inStrongLinks ? 'text-accent' : 'text-muted-foreground'}>
                          {regimeSensitivity.inStrongLinks ? 'Yes' : 'No'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">
                          {isCompareActive && compareRegime !== 'None' ? `${activeRegime} vs ${compareRegime}` : 'Under stress'}
                        </span>
                        <span className={`font-medium ${
                          (isCompareActive && selectedNodeCompare ? selectedNodeCompare.pairEdgeBehavior : regimeSensitivity.edgeBehavior) === 'gains' ? 'text-destructive' :
                          (isCompareActive && selectedNodeCompare ? selectedNodeCompare.pairEdgeBehavior : regimeSensitivity.edgeBehavior) === 'loses' ? 'text-accent' : 'text-muted-foreground'
                        }`}>
                          {(isCompareActive && selectedNodeCompare ? selectedNodeCompare.pairEdgeBehavior : regimeSensitivity.edgeBehavior) === 'gains' ? 'Gains edges' :
                           (isCompareActive && selectedNodeCompare ? selectedNodeCompare.pairEdgeBehavior : regimeSensitivity.edgeBehavior) === 'loses' ? 'Loses edges' : 'Stable'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Compare Across Regimes */}
                {isCompareActive && selectedNodeCompare && (
                  <div className="mt-3">
                    <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2 flex items-center gap-1.5">
                      <GitCompare className="w-3 h-3" />
                      Compare Across Regimes
                    </h4>
                    <div className="bg-secondary/60 rounded-lg p-3 space-y-2 text-[11px]">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Edges in {activeRegime}</span>
                        <span className="text-primary font-mono">{selectedNodeCompare.primaryEdges}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Edges in {compareRegime}</span>
                        <span className="text-accent font-mono">{selectedNodeCompare.compareEdges}</span>
                      </div>
                      {selectedNodeCompare.newNeighbors.length > 0 && (
                        <div>
                          <span className="text-muted-foreground text-[10px]">New in {compareRegime}:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {selectedNodeCompare.newNeighbors.slice(0, 4).map(n => (
                              <span key={n} className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-accent/10 text-accent">{n}</span>
                            ))}
                          </div>
                        </div>
                      )}
                      {selectedNodeCompare.lostNeighbors.length > 0 && (
                        <div>
                          <span className="text-muted-foreground text-[10px]">Lost in {compareRegime}:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {selectedNodeCompare.lostNeighbors.slice(0, 4).map(n => (
                              <span key={n} className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-destructive/10 text-destructive">{n}</span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Quick Actions */}
                <div className="mt-3">
                  <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2">Quick Actions</h4>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { label: 'Focus Neighborhood', icon: Crosshair },
                      { label: 'Highlight Incoming', icon: ArrowDownLeft },
                      { label: 'Highlight Outgoing', icon: ArrowUpRight },
                      { label: 'Center Graph', icon: Eye },
                    ].map(a => (
                      <button key={a.label}
                        className="flex items-center gap-1.5 px-2.5 py-2 rounded-lg bg-secondary/60 hover:bg-secondary text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                      >
                        <a.icon className="w-3 h-3" />
                        {a.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Current view info */}
                <div className="mt-5 flex items-center gap-2">
                  <Info className="w-3 h-3 text-muted-foreground shrink-0" />
                  <span className="text-[10px] text-muted-foreground">
                    Showing <span className="text-primary font-medium">{activeRegime}</span> regime · <span className="text-primary font-medium">{graphViews.find(v => v.id === graphView)?.label}</span>
                    {isCompareActive && <> · vs <span className="text-accent font-medium">{compareRegime}</span></>}
                  </span>
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="w-[280px] glass-strong border-l border-border flex items-center justify-center p-6 shrink-0"
            >
              <div className="text-center">
                <div className="w-12 h-12 rounded-xl bg-secondary/60 flex items-center justify-center mx-auto mb-3">
                  <Search className="w-5 h-5 text-muted-foreground" />
                </div>
                <p className="text-sm text-muted-foreground mb-1">No variable selected</p>
                <p className="text-[11px] text-muted-foreground/60 leading-relaxed">
                  Select a variable to inspect its drivers, dependents, and regime behavior.
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Lower Analysis Section (hover popup) ── */}
      <div
        className="relative shrink-0"
        onMouseEnter={() => setShowAnalysis(true)}
        onMouseLeave={() => setShowAnalysis(false)}
      >
        {/* Hover trigger strip */}
        <div className="border-t border-border bg-card/80 backdrop-blur flex items-center justify-center gap-2 py-1 cursor-pointer">
          <ChevronUp className={`w-3 h-3 text-muted-foreground transition-transform ${showAnalysis ? 'rotate-180' : ''}`} />
          <span className="text-[10px] uppercase tracking-widest text-muted-foreground">Analysis Workspace</span>
          <ChevronUp className={`w-3 h-3 text-muted-foreground transition-transform ${showAnalysis ? 'rotate-180' : ''}`} />
        </div>

        {/* Popup panel */}
        <AnimatePresence>
          {showAnalysis && (
            <motion.div
              initial={{ y: 10, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: 10, opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="absolute bottom-full left-0 right-0 border-t border-border bg-card shadow-lg z-50 overflow-x-auto"
            >
              <div className="px-4 py-2.5">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-2.5">
                  {/* Top Changed Relationships */}
                  {isCompareActive && compareAnalysis && (
                    <div className="bg-secondary/40 rounded-lg p-2.5">
                      <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1.5 flex items-center gap-1">
                        <GitCompare className="w-3 h-3" />
                        Top Changed Relationships
                      </h4>
                      <div className="space-y-2">
                        {compareAnalysis.added.slice(0, 3).map((e, i) => (
                          <div key={i} className="flex items-center justify-between text-[11px]">
                            <div className="flex items-center gap-1">
                              <Plus className="w-3 h-3 text-accent" />
                              <span className="font-mono text-foreground">{e.source}</span>
                              <span className="text-muted-foreground">→</span>
                              <span className="font-mono text-foreground">{e.target}</span>
                            </div>
                            <span className="text-accent text-[10px]">new</span>
                          </div>
                        ))}
                        {compareAnalysis.removed.slice(0, 2).map((e, i) => (
                          <div key={`r-${i}`} className="flex items-center justify-between text-[11px]">
                            <div className="flex items-center gap-1">
                              <Minus className="w-3 h-3 text-destructive" />
                              <span className="font-mono text-foreground/60">{e.source}</span>
                              <span className="text-muted-foreground">→</span>
                              <span className="font-mono text-foreground/60">{e.target}</span>
                            </div>
                            <span className="text-destructive text-[10px]">removed</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Most Stress-Sensitive Variables */}
                  <div className="bg-secondary/40 rounded-lg p-2.5">
                    <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1.5 flex items-center gap-1">
                      <Zap className="w-3 h-3" />
                      Most Stress-Sensitive Variables
                    </h4>
                    <div className="space-y-1.5">
                      {stressSensitiveVars.slice(0, 5).map((v, i) => (
                        <div key={i} className="flex items-center justify-between text-[11px]">
                          <span className="font-mono text-foreground">{v.id}</span>
                          <div className="flex items-center gap-2">
                            <span className="text-muted-foreground text-[10px]">{v.calm}→{v.stress}</span>
                            <span className={`font-mono text-[10px] ${v.delta > 0 ? 'text-destructive' : 'text-accent'}`}>
                              {v.delta > 0 ? '+' : ''}{v.delta}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Transmission Paths */}
                  <div className="bg-secondary/40 rounded-lg p-2.5">
                    <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1.5 flex items-center gap-1">
                      <Route className="w-3 h-3" />
                      Transmission Paths
                      {selectedNode && <span className="text-primary ml-1">· {selectedNode}</span>}
                    </h4>
                    {computedTransmissionPaths.length > 0 ? (
                      <div className="space-y-2.5">
                        {computedTransmissionPaths.slice(0, 5).map((tp, i) => (
                          <div key={i} className="group">
                            <div className="flex items-center gap-1 text-[11px] mb-0.5">
                              {tp.chain.map((node, j) => (
                                <span key={j} className="flex items-center gap-1">
                                  <span className={`font-mono ${node === selectedNode ? 'text-primary font-semibold' : 'text-foreground'}`}>{node}</span>
                                  {j < tp.chain.length - 1 && <ArrowRight className="w-2.5 h-2.5 text-muted-foreground" />}
                                </span>
                              ))}
                              <span className="ml-auto font-mono text-[10px] text-muted-foreground">{tp.strength.toFixed(2)}</span>
                            </div>
                            <p className="text-[10px] text-muted-foreground/70">{tp.interpretation}</p>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-[11px] text-muted-foreground/60 italic">
                        {graphView === 'transmission' ? 'No paths found for current filters.' : 'Switch to Transmission Paths view to discover directed chains.'}
                      </p>
                    )}
                  </div>

                  {/* Regime Comparison Summary */}
                  {isCompareActive && compareAnalysis ? (
                    <div className="bg-secondary/40 rounded-lg p-2.5">
                      <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-3">
                        Regime Comparison Summary
                      </h4>
                      <div className="space-y-2 text-[11px]">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Edges in {activeRegime}</span>
                          <span className="font-mono text-foreground">{compareAnalysis.primaryEdgeCount}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Edges in {compareRegime}</span>
                          <span className="font-mono text-foreground">{compareAnalysis.compareEdgeCount}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Edges added</span>
                          <span className="font-mono text-accent">{compareAnalysis.added.length}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Edges removed</span>
                          <span className="font-mono text-destructive">{compareAnalysis.removed.length}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Avg edge weight shift</span>
                          <span className={`font-mono ${compareAnalysis.avgEdgeWeightShift > 0 ? 'text-accent' : 'text-destructive'}`}>
                            {compareAnalysis.avgEdgeWeightShift > 0 ? '+' : ''}{compareAnalysis.avgEdgeWeightShift.toFixed(2)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Strengthened</span>
                          <span className="font-mono text-foreground">{compareAnalysis.strengthened.length}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Weakened</span>
                          <span className="font-mono text-foreground">{compareAnalysis.weakened.length}</span>
                        </div>
                      </div>
                    </div>
                  ) : (
                    /* Method Notes when not comparing */
                    <div className="bg-secondary/40 rounded-lg p-2.5">
                      <button
                        onClick={() => setMethodNotesOpen(!methodNotesOpen)}
                        className="flex items-center justify-between w-full text-left"
                      >
                        <h4 className="text-[10px] uppercase tracking-widest text-muted-foreground flex items-center gap-1.5">
                          <Info className="w-3 h-3" />
                          Method Notes
                        </h4>
                        {methodNotesOpen ? <ChevronDown className="w-3 h-3 text-muted-foreground" /> : <ChevronRight className="w-3 h-3 text-muted-foreground" />}
                      </button>
                      {methodNotesOpen && (
                        <div className="mt-3 space-y-2 text-[11px] text-muted-foreground">
                          <div>
                            <span className="text-foreground/70 font-medium">Discovery Graph</span>
                            <p>Broader learned relationship network with full recall. Includes weaker but potentially meaningful connections.</p>
                          </div>
                          <div>
                            <span className="text-foreground/70 font-medium">Strong Links</span>
                            <p>Cleaner filtered subset showing only stronger relationships (≥{(STRONG_LINK_MIN_WEIGHT * 100).toFixed(0)}% absolute weight).</p>
                          </div>
                          <div>
                            <span className="text-foreground/70 font-medium">Transmission Paths</span>
                            <p>Simplified route view through the network, highlighting key directional chains of influence.</p>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Legend ── */}
      <div className="glass border-t border-border flex flex-wrap items-center justify-center gap-x-5 gap-y-1.5 py-1.5 px-4 shrink-0">
        {Object.entries(categoryColors).map(([cat, color]) => (
          <div key={cat} className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-[10px] text-muted-foreground capitalize">{cat.replace('-', ' ')}</span>
          </div>
        ))}
        <div className="w-px h-3 bg-border mx-1" />
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5 bg-muted-foreground/30 rounded" />
          <span className="text-[10px] text-muted-foreground">Solid = direct</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5 border-t border-dashed border-muted-foreground/40" />
          <span className="text-[10px] text-muted-foreground">Dashed = lagged</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5 bg-muted-foreground/20 rounded" />
          <div className="w-1.5 h-0.5 bg-muted-foreground/40 rounded" />
          <span className="text-[10px] text-muted-foreground">Thicker = stronger</span>
        </div>
        {isCompareActive && (
          <>
            <div className="w-px h-3 bg-border mx-1" />
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-primary" />
              <span className="text-[10px] text-primary">{activeRegime}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-accent" />
              <span className="text-[10px] text-accent">{compareRegime}</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default CausalGraph;
