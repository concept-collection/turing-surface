import type { ColormapFunc } from './colormaps.ts';

/** Compact numeric label: 3 significant digits, trailing zeros trimmed. */
export const fmtValue = (v: number): string =>
  Number.isFinite(v) ? v.toPrecision(3).replace(/\.?0+$/, '') : '—';

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
