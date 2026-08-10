/**
 * Live Δ(t) chart for a comparison study: one canvas, one line per
 * (variant, species) pair, sharing a log-scaled y-axis and an x-axis in
 * model time `t`. A variant's lines are colored with that variant's own
 * accent color (the same color shown on the left edge of its row of
 * panels); species are told apart within one color by line style (solid,
 * dashed, dotted, …), matched by the legend's dash swatches.
 *
 * Deliberately dumb — no library, just a canvas redrawn from scratch on every
 * push(). History is unbounded (a study is a "does it converge" question, and
 * capping the window would hide exactly the slow drift that question is
 * about); only the running log-min/max is maintained incrementally so a
 * redraw stays O(points on screen), not O(history) beyond what it already
 * draws.
 */

import { fmtValue } from './colorbar.ts';

export interface ErrorChartRow {
  label: string;
  color: string;
}

/** Floor for the log scale — well below any error this app ever measures,
 *  just far enough that log10 never sees zero or a negative. */
const MIN_LOG_VALUE = 1e-12;

/** Canvas line-dash patterns by species index — solid, dashed, dotted,
 *  repeating for a fourth-plus species (no shipped model has one). Mirrored
 *  in index.html's `.cmp-chart-dash-{k}` border-style classes for the
 *  legend swatches. */
const DASH_PATTERNS: number[][] = [[], [7, 4], [2, 3]];

const MARGIN_LEFT = 44;
const MARGIN_BOTTOM = 38;
const MARGIN_TOP = 8;
const MARGIN_RIGHT = 10;

export class ErrorChart {
  #container: HTMLElement;
  #species: string[];
  #rows: ErrorChartRow[];
  #canvas: HTMLCanvasElement | null = null;

  #ts: number[] = [];
  /** #errs[k][i] is one species' one row's history, same length as #ts. */
  #errs: number[][][] = [];
  /** Running log10 extent across every (species, row) pushed — one shared
   *  y-axis now, not one per species. */
  #minLog = Infinity;
  #maxLog = -Infinity;

  constructor(container: HTMLElement, species: string[], rows: ErrorChartRow[]) {
    this.#container = container;
    this.#species = species;
    this.#rows = rows;
    this.#resetSeries();
    this.#build();
  }

  #resetSeries(): void {
    this.#ts = [];
    this.#errs = this.#species.map(() => this.#rows.map(() => []));
    this.#minLog = Infinity;
    this.#maxLog = -Infinity;
  }

  #build(): void {
    this.#container.replaceChildren();
    this.#canvas = null;
    if (this.#rows.length === 0 || this.#species.length === 0) return;

    const legend = document.createElement('div');
    legend.className = 'cmp-chart-legend';

    const variantGroup = document.createElement('div');
    variantGroup.className = 'cmp-chart-legend-group';
    for (const r of this.#rows) {
      const item = document.createElement('span');
      item.className = 'cmp-chart-legend-item';
      const swatch = document.createElement('i');
      swatch.style.background = r.color;
      item.append(swatch, document.createTextNode(r.label));
      variantGroup.append(item);
    }

    const speciesGroup = document.createElement('div');
    speciesGroup.className = 'cmp-chart-legend-group cmp-chart-legend-species';
    this.#species.forEach((name, k) => {
      const item = document.createElement('span');
      item.className = 'cmp-chart-legend-item';
      const swatch = document.createElement('i');
      swatch.className = `cmp-chart-dash cmp-chart-dash-${k % DASH_PATTERNS.length}`;
      item.append(swatch, document.createTextNode(name));
      speciesGroup.append(item);
    });

    legend.append(variantGroup, speciesGroup);
    this.#container.append(legend);

    const canvas = document.createElement('canvas');
    canvas.className = 'cmp-chart-canvas';
    this.#container.append(canvas);
    this.#canvas = canvas;
  }

  /** One frame's sample: `t` shared by every row, `perRowErr[i][k]` the
   *  relative-L2 error of row i's species k against the reference. Rows here
   *  are exactly the ones passed to the constructor, in the same order — the
   *  reference row is never included (its Δ against itself is always 0). */
  push(t: number, perRowErr: number[][]): void {
    this.#ts.push(t);
    for (let k = 0; k < this.#species.length; k++) {
      for (let i = 0; i < this.#rows.length; i++) {
        const v = perRowErr[i]?.[k];
        this.#errs[k][i].push(v === undefined ? NaN : v);
        if (Number.isFinite(v) && v! > 0) {
          const lv = Math.log10(v!);
          if (lv < this.#minLog) this.#minLog = lv;
          if (lv > this.#maxLog) this.#maxLog = lv;
        }
      }
    }
    this.#draw();
  }

  /** Back to no history — called wherever the study's clock itself resets
   *  (restart, reseed), so the chart never shows a curve spanning a rewind. */
  reset(): void {
    this.#resetSeries();
    this.#draw();
  }

  /** Redraw with the data as it stands — for a container resize, where
   *  nothing new has been measured but the canvas backing buffer has to
   *  change to match. */
  redraw(): void {
    this.#draw();
  }

  dispose(): void {
    this.#container.replaceChildren();
    this.#canvas = null;
  }

  #draw(): void {
    const canvas = this.#canvas;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.max(1, rect.width);
    const cssH = Math.max(1, rect.height || 240);
    const pxW = Math.max(1, Math.round(cssW * dpr));
    const pxH = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== pxW) canvas.width = pxW;
    if (canvas.height !== pxH) canvas.height = pxH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    // Draw in CSS-pixel coordinates throughout; the transform alone accounts
    // for devicePixelRatio, rather than scaling every margin/font by hand.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const n = this.#ts.length;
    if (n < 2) return;

    const rawLo = this.#minLog;
    const rawHi = this.#maxLog;
    if (!Number.isFinite(rawLo) || !Number.isFinite(rawHi)) return;
    let lo = rawLo;
    let hi = rawHi;
    if (hi - lo < 1e-6) {
      lo -= 0.5;
      hi += 0.5;
    }
    const pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;

    const style = getComputedStyle(document.documentElement);
    const inkColor = style.getPropertyValue('--ink-2').trim() || '#666';
    const gridColor = style.getPropertyValue('--line').trim() || '#ccc';

    const plotX = MARGIN_LEFT;
    const plotY = MARGIN_TOP;
    const plotW = Math.max(1, cssW - MARGIN_LEFT - MARGIN_RIGHT);
    const plotH = Math.max(1, cssH - MARGIN_TOP - MARGIN_BOTTOM);

    const t0 = this.#ts[0];
    const t1 = this.#ts[n - 1];
    const tSpan = t1 - t0 || 1;
    const xAt = (t: number): number => plotX + ((t - t0) / tSpan) * plotW;
    const yAt = (v: number): number => {
      const lv = Math.max(lo, Math.min(hi, Math.log10(Math.max(v, MIN_LOG_VALUE))));
      return plotY + (1 - (lv - lo) / (hi - lo)) * plotH;
    };

    ctx.font = '11px sans-serif';

    // ---- y-axis: decade gridlines + ticks + labels -----------------------
    const yTicks = decadeTicks(lo, hi, rawLo, rawHi);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const tick of yTicks) {
      const y = yAt(tick.value);
      ctx.strokeStyle = gridColor;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plotX, y);
      ctx.lineTo(plotX + plotW, y);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.moveTo(plotX - 4, y);
      ctx.lineTo(plotX, y);
      ctx.stroke();
      ctx.fillStyle = inkColor;
      ctx.fillText(tick.label, plotX - 6, y);
    }

    // ---- x-axis: ticks + labels --------------------------------------------
    const xTicks = evenTicks(t0, t1, 5);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.strokeStyle = gridColor;
    for (const t of xTicks) {
      const x = xAt(t);
      ctx.beginPath();
      ctx.moveTo(x, plotY + plotH);
      ctx.lineTo(x, plotY + plotH + 4);
      ctx.stroke();
      ctx.fillStyle = inkColor;
      ctx.fillText(fmtValue(t), x, plotY + plotH + 6);
    }

    // ---- axis frame ---------------------------------------------------------
    ctx.strokeStyle = gridColor;
    ctx.beginPath();
    ctx.moveTo(plotX, plotY);
    ctx.lineTo(plotX, plotY + plotH);
    ctx.lineTo(plotX + plotW, plotY + plotH);
    ctx.stroke();

    // ---- axis titles ---------------------------------------------------------
    ctx.fillStyle = inkColor;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('t', plotX + plotW / 2, cssH - 2);

    ctx.save();
    ctx.translate(12, plotY + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('relative L2 error', 0, 0);
    ctx.restore();

    // ---- the data: one line per (species, row) -----------------------------
    for (let k = 0; k < this.#species.length; k++) {
      const dash = DASH_PATTERNS[k % DASH_PATTERNS.length];
      for (let i = 0; i < this.#rows.length; i++) {
        const series = this.#errs[k][i];
        ctx.setLineDash(dash);
        ctx.strokeStyle = this.#rows[i].color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        let started = false;
        for (let j = 0; j < n; j++) {
          const v = series[j];
          // A non-finite or non-positive sample (a diverged row, or a norm
          // with zero denominator) breaks the line rather than drawing a
          // spurious segment through a point that has no place on a log axis.
          if (!Number.isFinite(v) || v <= 0) {
            started = false;
            continue;
          }
          const x = xAt(this.#ts[j]);
          const y = yAt(v);
          if (started) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
          started = true;
        }
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
  }
}

/**
 * Y-axis ticks: one per decade spanned by [lo, hi] (the padded plotting
 * range), labeled `1e{n}`. If the visible range covers less than a full
 * decade — a study that hasn't had time to spread out yet — decade ticks
 * would give zero or one of them, so fall back to two ticks at the actual
 * (unpadded) data extent instead, labeled with their real value.
 */
function decadeTicks(
  lo: number,
  hi: number,
  rawLo: number,
  rawHi: number,
): { value: number; label: string }[] {
  const start = Math.ceil(lo);
  const end = Math.floor(hi);
  const decades: number[] = [];
  for (let d = start; d <= end; d++) decades.push(d);
  if (decades.length >= 2) {
    return decades.map((d) => ({
      value: 10 ** d,
      label: (10 ** d).toExponential(0).replace('e+', 'e'),
    }));
  }
  const loVal = 10 ** rawLo;
  const hiVal = 10 ** rawHi;
  if (hiVal <= loVal) return [{ value: loVal, label: fmtValue(loVal) }];
  return [
    { value: loVal, label: fmtValue(loVal) },
    { value: hiVal, label: fmtValue(hiVal) },
  ];
}

/** `count` evenly spaced ticks between t0 and t1 inclusive; just t0 if the
 *  span is degenerate (a single point pushed so far). */
function evenTicks(t0: number, t1: number, count: number): number[] {
  if (t1 <= t0) return [t0];
  return Array.from({ length: count }, (_, i) => t0 + (i * (t1 - t0)) / (count - 1));
}
