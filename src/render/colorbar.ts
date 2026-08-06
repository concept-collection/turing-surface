import type { ColormapFunc } from './colormaps.ts';

/** Compact numeric label: 3 significant digits, trailing zeros trimmed. */
export const fmtValue = (v: number): string =>
  Number.isFinite(v) ? v.toPrecision(3).replace(/\.?0+$/, '') : '—';

/**
 * Smallest span the colormap may be stretched across, relative to the field's
 * own magnitude.
 *
 * Set from measurement, not taste. A constant field analysed and re-synthesized
 * in fp32 comes back constant only to
 *
 *   lmax  63:  2.9e-5 relative      lmax 127:  9.6e-5      lmax 255:  2.4e-4
 *
 * and the residue is not white noise — it is concentrated in a few rings at the
 * poles (74x the equatorial level at lmax 63, 1710x at lmax 255), because what
 * survives the analysis is high-degree m = 0 content whose Legendre functions
 * all peak at the poles *and add in phase there*. The same round trip in f64 is
 * 2.4e-8 and flat, so this is fp32, not the algorithm.
 *
 * A floor of 1e-2 puts the worst of that (about 5e-4 of span at lmax 255) into
 * roughly 5% of the colormap rather than all of it, while the variation these
 * models actually carry — a few percent of the field's magnitude and up — is
 * left alone entirely.
 */
const RANGE_FLOOR_REL = 1e-2;
/** And an absolute floor, for a field whose magnitude is itself near zero. */
const RANGE_FLOOR_ABS = 1e-9;

/**
 * Widen a value range so that a field which is uniform to numerical precision
 * is drawn as uniform.
 *
 * Scaling the colormap to a field's own extremes gives full contrast to
 * whatever variation it has — including none. Schnakenberg's `v` at t = 0 is
 * literally constant (`vs * ones(...)`), so its extremes are set purely by the
 * roundoff described above; painting that across the whole colormap produces a
 * vivid pole-capped picture that reads as structure, and since the residue
 * belongs to the grid, two runs at different lmax produce two entirely
 * different pictures of the same constant — which looks exactly like a broken
 * initial condition, and is not one.
 *
 * A floor rather than an "is this field constant?" test, so nothing ever jumps:
 * a real pattern growing up through the floor hands the range over from the
 * floor to its own data gradually, and once it is any larger than roundoff the
 * floor has no effect at all.
 */
export function floorRange(lo: number, hi: number): { lo: number; hi: number } {
  const minSpan = Math.max(
    RANGE_FLOOR_ABS,
    RANGE_FLOOR_REL * Math.max(Math.abs(lo), Math.abs(hi)),
  );
  if (hi - lo >= minSpan) return { lo, hi };
  const mid = (lo + hi) / 2;
  return { lo: mid - minSpan / 2, hi: mid + minSpan / 2 };
}

/** Vertical colorbar drawn on a small canvas, with min/max labels. */
export class Colorbar {
  #canvas: HTMLCanvasElement;
  #minLabel: HTMLElement;
  #maxLabel: HTMLElement;

  constructor(container: HTMLElement) {
    container.classList.add('colorbar');
    this.#maxLabel = document.createElement('div');
    this.#maxLabel.className = 'colorbar-label';
    this.#canvas = document.createElement('canvas');
    this.#canvas.width = 12;
    this.#canvas.height = 160;
    this.#minLabel = document.createElement('div');
    this.#minLabel.className = 'colorbar-label';
    container.append(this.#maxLabel, this.#canvas, this.#minLabel);
  }

  update(cmap: ColormapFunc, vmin: number, vmax: number): void {
    const ctx = this.#canvas.getContext('2d');
    if (!ctx) return;
    const h = this.#canvas.height;
    for (let y = 0; y < h; y++) {
      const t = 1 - y / (h - 1);
      const [r, g, b] = cmap(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(0, y, this.#canvas.width, 1);
    }
    this.#maxLabel.textContent = fmtValue(vmax);
    this.#minLabel.textContent = fmtValue(vmin);
  }
}
