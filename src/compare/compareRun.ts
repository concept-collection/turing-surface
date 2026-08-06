/**
 * Several solver settings, one problem, one clock.
 *
 * A convergence study of the knobs that decide how well the implicit solve is
 * resolved — `niter`, `lmax`, `dt` — run side by side so the answer to "does it
 * matter?" is visible rather than argued. Every variant is its own
 * `ModelSession` (both `niter` and `lmax` are structural: they change the
 * compiled step and the grid), and what makes the set a comparison rather than
 * a collection is three things they are forced to share:
 *
 * - **One initial condition.** Band-limited at the coarsest variant's lmax and
 *   evaluated on each variant's own grid, so every session starts from the same
 *   *function* rather than from the same random seed — see sharedStart.ts for
 *   why the seed alone is not enough.
 *
 * - **One clock.** Variants differ in dt only by an integer power-of-two
 *   divisor, and a frame advances each of them by `frameSteps * dtDiv` steps.
 *   Every variant therefore lands on exactly the same model time at the end of
 *   every frame, having taken a different number of steps to get there. Nothing
 *   is ever compared across a time offset.
 *
 * - **One grid to look at.** Each session's *display* plan is pointed at a
 *   common grid (ModelSession.setDisplayGrid), which is exact evaluation rather
 *   than resampling because the state is band-limited. So the fields come back
 *   directly comparable point by point, one mesh topology serves every panel,
 *   and the difference norm is an ordinary weighted sum.
 *
 * What is *not* shared is the surface: each variant carries the geometry
 * band-limited at its own lmax, and renders the surface it actually solves on.
 */
import { ModelSession } from '../mgpu/session.ts';
import type { MModel, Params } from '../mgpu/registry.ts';
import type { MGeometry } from '../geom/registry.ts';
import {
  buildTopology,
  fillFieldValues,
  fillPositions,
  fillColors,
  type SphereMeshTopology,
} from '../render/sphereMesh.ts';
import { SphereScene } from '../render/SphereScene.ts';
import { colormaps } from '../render/colormaps.ts';
import { fmtValue, floorRange } from '../render/colorbar.ts';
import { sharedNoise } from './sharedStart.ts';
import { variantLabel, VARIANT_COLORS, type Variant } from './variants.ts';

/**
 * Latitudes of the shared display grid. 256 is the same target the single-run
 * view uses for 'auto' oversampling, and for the same reason — beyond it a
 * finer mesh costs vertices without showing anything.
 *
 * Here it is a ceiling as well as a target, in two directions. At lmax 255 the
 * solver grid is finer than this, so the panels sample the (exact) state more
 * coarsely than the solver carries it; and past a handful of panels the mesh is
 * paid for once per panel, in vertices, normals and a WebGL context each, so it
 * halves. Both are display choices, both are reported in the status line, and
 * neither touches the difference norm's meaning: that is computed on this same
 * grid for every variant, so it stays a consistent comparison whatever the grid.
 */
const RENDER_NLAT = 256;
const RENDER_NLAT_CROWDED = 128;
const CROWDED_PANELS = 6;

/** See main.ts's DISPATCH_BUDGET — the same watchdog argument, per variant. */
const DISPATCH_BUDGET = 1000;
const STEPS_PER_FRAME_BASE = 4;

export interface CompareOptions {
  device: GPUDevice;
  model: MModel;
  /** The model's parameters, with `dt` read as the *base* timestep that each
   *  variant's dtDiv divides. */
  params: Params;
  source: string;
  geometry: MGeometry;
  geometryParams: Params;
  geometrySource: string;
  variants: Variant[];
  /** Index into `variants` of the run everything else is measured against. */
  reference: number;
  seed: number;
  morph: number;
  colormapName: () => string;
  /** Where the variant grid goes (the app's #panels). */
  container: HTMLElement;
  /** Progress and, afterwards, the standing description of the study. */
  onStatus: (html: string) => void;
}

interface Row {
  variant: Variant;
  session: ModelSession;
  color: string;
  /** Surface coordinates on the shared render grid — this variant's own. */
  coords: Float32Array;
  posBuf: Float32Array;
  scenes: SphereScene[];
  valueBufs: Float32Array[];
  colorBufs: Float32Array[];
  /** Fields read this frame, one per species, on the shared grid. */
  fields: Float32Array[];
  /** Relative difference from the reference, one per species. */
  err: number[];
  /** False once any species has left the floating-point numbers — the shape a
   *  variant outside the convergence radius eventually fails in. Such a row is
   *  never used to scale a column, and its label says so. */
  healthy: boolean;
  statEl: HTMLElement;
}

export class CompareRun {
  #opts: CompareOptions;
  #rows: Row[] = [];
  #topo: SphereMeshTopology;
  /** Quadrature weight per grid point of the shared grid, for the L2 norm. */
  #weights: Float64Array;
  #rangeBars: { fill: (lo: number, hi: number) => void }[] = [];
  /** Smoothed color range per species, shared by every variant so the panels
   *  in a column are directly comparable by eye and not just by number. */
  #ranges: { lo: number; hi: number }[] = [];
  #resizeObs: ResizeObserver | null = null;

  #running = false;
  #pumping = false;
  #disposed = false;
  #morph: number;
  /** Base steps per frame; variant i takes this times its dtDiv. */
  #frameSteps = STEPS_PER_FRAME_BASE;
  /** Model time all variants are at — one number, by construction. */
  #t = 0;
  #frameMs = 0;
  #note: string;

  private constructor(init: {
    opts: CompareOptions;
    rows: Row[];
    topo: SphereMeshTopology;
    weights: Float64Array;
    rangeBars: { fill: (lo: number, hi: number) => void }[];
    frameSteps: number;
    note: string;
  }) {
    this.#opts = init.opts;
    this.#rows = init.rows;
    this.#topo = init.topo;
    this.#weights = init.weights;
    this.#rangeBars = init.rangeBars;
    this.#frameSteps = init.frameSteps;
    this.#note = init.note;
    this.#morph = init.opts.morph;
    this.#ranges = init.opts.model.species.map(() => ({ lo: NaN, hi: NaN }));
  }

  get variants(): Variant[] {
    return this.#rows.map((r) => r.variant);
  }

  /** The variant everything else is measured against — the one whose numbers
   *  stand on their own, so the one the app quotes when it has to quote one. */
  get referenceSession(): ModelSession | null {
    return this.#rows[this.#opts.reference]?.session ?? null;
  }

  get referenceIndex(): number {
    return this.#opts.reference;
  }

  /** The base timestep a variant's dtDiv divides. */
  static baseDt(params: Params): number {
    return params.dt ?? 0;
  }

  static async create(opts: CompareOptions): Promise<CompareRun> {
    const { device, model, variants } = opts;
    const baseDt = CompareRun.baseDt(opts.params);
    const showDt = variants.some((v) => v.dtDiv !== variants[0].dtDiv);
    const sessions: ModelSession[] = [];
    // Scenes own a WebGL context and an animation frame each, so a failure
    // after the grid is up has to take them down explicitly — removing their
    // canvases from the DOM would leave both running.
    let built: Row[] = [];

    try {
      for (let i = 0; i < variants.length; i++) {
        const v = variants[i];
        opts.onStatus(
          `compiling ${i + 1}/${variants.length} — ${variantLabel(v, showDt)} ` +
            `(a solve iteration is ~15 kernels per species, and there is no ` +
            `pipeline cache across sessions)`,
        );
        // Yield, so the status actually paints before the compile blocks.
        await new Promise<number>(requestAnimationFrame);
        sessions.push(
          await ModelSession.create({
            device,
            model,
            params: { ...opts.params, dt: baseDt / v.dtDiv },
            lmax: v.lmax,
            source: opts.source,
            geometry: opts.geometry,
            geometryParams: opts.geometryParams,
            geometrySource: opts.geometrySource,
            niter: v.niter,
          }),
        );
      }

      // ---- the shared display grid ----------------------------------------
      const maxLmax = Math.max(...variants.map((v) => v.lmax));
      const panels = variants.length * model.species.length;
      const target = panels > CROWDED_PANELS ? RENDER_NLAT_CROWDED : RENDER_NLAT;
      // Never below what the finest band needs to be representable at all
      // (ShtPlan requires nlat > lmax), whatever the panel count says.
      const nlat = Math.max(target, 2 * Math.ceil((maxLmax + 2) / 2));
      let nphi = 1;
      while (nphi < Math.max(2 * nlat, 2 * maxLmax + 1)) nphi *= 2;
      for (const s of sessions) await s.setDisplayGrid(nlat, nphi);

      // ---- one initial condition, on every grid ---------------------------
      opts.onStatus('seeding all variants from one band-limited perturbation…');
      const noise = await sharedNoise(sessions, model.seedAmp, opts.seed);
      sessions.forEach((s, i) => s.seedWith(noise[i]));

      // ---- the mesh, shared; the surface, per variant ---------------------
      const view = sessions[0].viewSht;
      const phi = new Float64Array(nphi);
      for (let j = 0; j < nphi; j++) phi[j] = (2 * Math.PI * j) / nphi;
      const topo = buildTopology(view.cosTheta, phi);
      // Gauss weights carry the sin(theta) of the area element; the constant
      // 2*pi/nphi is common to every point and cancels in the relative norm.
      const weights = new Float64Array(nlat * nphi);
      for (let i = 0; i < nlat; i++) {
        for (let j = 0; j < nphi; j++) weights[i * nphi + j] = view.gaussWeights[i];
      }

      // ---- how many steps a frame may submit ------------------------------
      // Per variant: its own unrolled step size times its dtDiv, since a ÷K
      // variant takes K times as many steps to reach the same time.
      let frameSteps = STEPS_PER_FRAME_BASE;
      const ops: number[] = [];
      for (let i = 0; i < sessions.length; i++) {
        const n = Math.max(1, sessions[i].describe().step.length);
        ops.push(n);
        frameSteps = Math.min(
          frameSteps,
          Math.max(1, Math.floor(DISPATCH_BUDGET / (n * variants[i].dtDiv))),
        );
      }
      frameSteps = Math.max(1, frameSteps);

      // ---- the grid of panels ---------------------------------------------
      const { rows, rangeBars } = await buildGrid(opts, sessions, topo, showDt);
      built = rows;

      const solverGrid = sessions.map((s) => `${s.cfg.nlat}×${s.cfg.nphi}`);
      const note =
        `${variants.length} variants · display grid ${nlat}×${nphi}` +
        (sessions.some((s) => s.cfg.nlat > nlat)
          ? ` (below the finest solver grid ${solverGrid[solverGrid.length - 1]} — display only)`
          : '') +
        ` · ${frameSteps} base step${frameSteps === 1 ? '' : 's'}/frame` +
        ` · ops/step ${ops.join(', ')}`;

      const run = new CompareRun({
        opts, rows, topo, weights, rangeBars, frameSteps, note,
      });
      await run.draw();
      run.#observeResize();
      run.#status();
      return run;
    } catch (e) {
      for (const r of built) for (const s of r.scenes) s.dispose();
      for (const s of sessions) s.destroy();
      opts.container.replaceChildren();
      opts.container.classList.remove('compare');
      throw e;
    }
  }

  // ------------------------------------------------------------------ state
  setRunning(next: boolean): void {
    this.#running = next;
    if (next) void this.#pump();
  }

  get running(): boolean {
    return this.#running;
  }

  /** Re-seed every variant from one new shared perturbation. */
  async reseed(seed: number): Promise<void> {
    const wasRunning = this.#running;
    this.#running = false;
    while (this.#pumping) await nextFrame();
    if (this.#disposed) return;
    const noise = await sharedNoise(
      this.#rows.map((r) => r.session),
      this.#opts.model.seedAmp,
      seed,
    );
    if (this.#disposed) return;
    this.#rows.forEach((r, i) => r.session.seedWith(noise[i]));
    this.#t = 0;
    for (const r of this.#ranges) {
      r.lo = NaN;
      r.hi = NaN;
    }
    await this.draw();
    if (!this.#disposed && wasRunning) this.setRunning(true);
  }

  /** Model parameters changed. Each variant keeps its own dt. */
  setParams(params: Params): void {
    this.#opts.params = params;
    const baseDt = CompareRun.baseDt(params);
    for (const r of this.#rows) {
      r.session.setParams({ ...params, dt: baseDt / r.variant.dtDiv });
    }
  }

  setMorph(morph: number): void {
    this.#morph = morph;
    for (const r of this.#rows) {
      fillPositions(r.posBuf, r.coords, this.#topo, morph);
      for (const s of r.scenes) s.updatePositions(r.posBuf);
    }
  }

  resetView(): void {
    for (const r of this.#rows) for (const s of r.scenes) s.resetCamera();
  }

  dispose(): void {
    this.#disposed = true;
    this.#running = false;
    this.#resizeObs?.disconnect();
    this.#resizeObs = null;
    for (const r of this.#rows) {
      for (const s of r.scenes) s.dispose();
      r.session.destroy();
    }
    this.#rows = [];
    this.#opts.container.replaceChildren();
    this.#opts.container.classList.remove('compare');
  }

  // ----------------------------------------------------------------- drawing
  /**
   * One frame's readback: every variant's every species, on the shared grid.
   * Read first, then color — the range is shared down a column, so no panel can
   * be filled until the column's range is known.
   */
  async draw(): Promise<void> {
    if (this.#disposed) return;
    const species = this.#opts.model.species;
    // Sessions are independent, so their readbacks can be in flight together;
    // within one session they must not be (they share its staging buffers).
    await Promise.all(
      this.#rows.map(async (r) => {
        for (let k = 0; k < species.length; k++) {
          r.fields[k] = await r.session.readSpecies(k);
        }
      }),
    );
    if (this.#disposed) return;

    const cmap = colormaps[this.#opts.colormapName()] ?? colormaps.viridis;

    /**
     * What scales a column is the whole question, and it has three wrong
     * answers.
     *
     * Per panel is wrong: a range each rescales every variant to itself and
     * hides exactly the difference the grid exists to show. The union over
     * variants is wrong for the opposite reason: a variant outside the
     * iteration's convergence radius runs away to 1e20 and then to NaN, and a
     * union range rescales the *whole column* to it, flattening every panel to
     * one colour — which reads as "they all blew up" when only one did.
     *
     * The reference alone is wrong too, less obviously, and it is the case that
     * actually bites: outside the convergence radius *more* Richardson
     * iterations diverge *faster*, so the row that goes first is usually the
     * highest-niter one — which is the reference.
     *
     * So the column is scaled by whichever variant **reaches least far from
     * zero** — the least-blown-up one. That is a comparison between the rows,
     * not a threshold on any of them, and the distinction is the whole point:
     * any "is this value too big?" test has a window in which a diverging field
     * is still under the limit, and for as long as that window lasts it drags
     * the scale and flattens the grid, until it finally trips and everything
     * springs back. A comparison has no such window — a run-away only has to be
     * *larger* than a healthy row to stop setting the scale, which it is from
     * its first bad step, and it stays larger no matter how many other rows go
     * with it. One healthy variant is enough to keep the grid readable.
     *
     * The cost is a slight bias: among healthy variants the scale comes from
     * the one with the smallest peak, so the others clip by however much they
     * exceed it. They are approximations of the same solution, so that is a
     * fraction of a percent, and the alternative is a display that a single
     * divergence can take away.
     */
    const bounds = this.#rows.map((r) => species.map((_, k) => finiteRange(r.fields[k])));
    this.#rows.forEach((r, i) => {
      // A row with any non-finite value is out of the running entirely: its
      // finite entries are whatever survived, and no rank over them means much.
      r.healthy = species.every((_, k) => allFinite(r.fields[k]) && bounds[i][k] !== null);
    });

    for (let k = 0; k < species.length; k++) {
      const anchor = leastPeak(this.#rows.map((r, i) => (r.healthy ? bounds[i][k] : null)));
      const range = this.#ranges[k];
      if (anchor) {
        if (!Number.isFinite(range.lo)) {
          range.lo = anchor.lo;
          range.hi = anchor.hi;
        } else {
          // Smooth in both directions so the shading evolves gently as the
          // pattern grows, as the single-run view does.
          const a = 0.15;
          range.lo += a * (anchor.lo - range.lo);
          range.hi += a * (anchor.hi - range.hi);
        }
      }
      // With every row gone, the last good range is kept rather than replaced
      // by nothing: the panels freeze at a readable scale and the row labels
      // say what happened, instead of the grid going blank.
      if (!Number.isFinite(range.lo) || !Number.isFinite(range.hi)) continue;
      // The floor is applied to what is drawn, not to what is tracked, so it
      // never feeds back into the smoothing above.
      const shown = floorRange(range.lo, range.hi);
      this.#rangeBars[k]?.fill(shown.lo, shown.hi);
      for (const r of this.#rows) {
        fillFieldValues(r.valueBufs[k], r.fields[k], this.#topo);
        fillColors(r.colorBufs[k], r.valueBufs[k], shown.lo, shown.hi, cmap);
        r.scenes[k]?.updateColors(r.colorBufs[k]);
      }
    }

    this.#measureDifference();
    this.#updateRowStats();
  }

  /**
   * Relative L2 difference from the reference, per species, on the shared
   * grid. Weighted by the Gauss weights, so it is the norm on the parameter
   * sphere — not on the embedded surface, which would weight by the area
   * element. That makes it a consistent diagnostic across variants rather than
   * a physical quantity, which is all it is used for.
   */
  #measureDifference(): void {
    const ref = this.#rows[this.#opts.reference];
    if (!ref) return;
    const species = this.#opts.model.species;
    for (const r of this.#rows) {
      for (let k = 0; k < species.length; k++) {
        if (r === ref) {
          r.err[k] = 0;
          continue;
        }
        const a = r.fields[k];
        const b = ref.fields[k];
        if (!a || !b || a.length !== b.length) {
          r.err[k] = NaN;
          continue;
        }
        let num = 0;
        let den = 0;
        for (let i = 0; i < a.length; i++) {
          const w = this.#weights[i];
          const d = a[i] - b[i];
          num += w * d * d;
          den += w * b[i] * b[i];
        }
        r.err[k] = den > 0 ? Math.sqrt(num / den) : NaN;
      }
    }
  }

  /**
   * Each row's standing line: how many of its own steps it took to reach the
   * common time, and how far it is from the reference right now, per species.
   * Per species rather than a single worst-case number because the two are
   * genuinely different questions on a two-species model — the slow species is
   * usually the one that has converged and the fast one the one that has not.
   */
  #updateRowStats(): void {
    const species = this.#opts.model.species;
    const ref = this.#rows[this.#opts.reference];
    for (const r of this.#rows) {
      const per = species
        .map((s, k) => `${s} ${Number.isFinite(r.err[k]) ? r.err[k].toExponential(2) : '—'}`)
        .join('<br>');
      // Divergence is said, not implied. Scaled to a healthy row, a blown-up
      // variant is a flat saturated panel, which on its own is easy to misread
      // as a converged uniform state.
      const body = !r.healthy
        ? '<b class="cmp-diverged">diverged</b>'
        : r === ref
          ? '<b>reference</b>'
          : `Δ ${per}`;
      r.statEl.innerHTML = `${r.session.steps.toLocaleString()} steps<br>${body}`;
    }
  }

  #status(): void {
    this.#opts.onStatus(
      `<b>t = ${this.#t.toFixed(2)}</b> (same for every variant) · ` +
        (this.#frameMs > 0 ? `${this.#frameMs.toFixed(1)} ms/frame · ` : '') +
        this.#note,
    );
  }

  #observeResize(): void {
    const scenes = this.#rows.flatMap((r) => r.scenes);
    this.#resizeObs = new ResizeObserver(() => {
      for (const s of scenes) {
        const box = s.canvas.parentElement;
        if (box) s.resize(box.clientWidth, box.clientHeight);
      }
    });
    for (const s of scenes) {
      const box = s.canvas.parentElement;
      if (box) this.#resizeObs.observe(box);
    }
  }

  // -------------------------------------------------------------- the clock
  /**
   * One frame advances every variant by the *same model time*: `frameSteps`
   * base steps, which a ÷K variant covers in K times as many of its own. That
   * is the whole reason dt varies by an integer divisor — the alternative is
   * rounding each variant to the nearest step and comparing fields that are a
   * fraction of a timestep apart, which would show up as a difference and be
   * indistinguishable from a real one.
   */
  async #pump(): Promise<void> {
    if (this.#pumping) return;
    this.#pumping = true;
    try {
      while (this.#running && !this.#disposed) {
        const t0 = performance.now();
        for (const r of this.#rows) r.session.step(this.#frameSteps * r.variant.dtDiv);
        this.#t += this.#frameSteps * CompareRun.baseDt(this.#opts.params);
        await this.draw();
        if (this.#disposed) break;
        const dt = performance.now() - t0;
        this.#frameMs = this.#frameMs === 0 ? dt : this.#frameMs + 0.05 * (dt - this.#frameMs);
        this.#status();
        await nextFrame();
      }
      if (!this.#disposed) {
        await this.draw();
        this.#status();
      }
    } finally {
      this.#pumping = false;
    }
  }
}

const nextFrame = (): Promise<number> => new Promise(requestAnimationFrame);

/** Whether every entry is an ordinary number — false once a variant has left
 *  its convergence radius and saturated to infinity or NaN. */
function allFinite(f: Float32Array | undefined): boolean {
  if (!f) return false;
  for (let i = 0; i < f.length; i++) if (!Number.isFinite(f[i])) return false;
  return true;
}

type Bounds = { lo: number; hi: number };

/** How far a field reaches from zero — the one number the rows are ranked by
 *  when deciding which of them sets a column's scale. */
const peak = (b: Bounds): number => Math.max(Math.abs(b.lo), Math.abs(b.hi));

/** Whichever of the given bounds reaches least far from zero; null if none. */
function leastPeak(all: (Bounds | null)[]): Bounds | null {
  let best: Bounds | null = null;
  for (const b of all) {
    if (b !== null && (best === null || peak(b) < peak(best))) best = b;
  }
  return best;
}

/** Min and max over the finite entries only; null when there are none. */
function finiteRange(f: Float32Array | undefined): { lo: number; hi: number } | null {
  if (!f) return null;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < f.length; i++) {
    const v = f[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? { lo, hi } : null;
}

/**
 * The DOM: a header row naming each species and carrying that column's shared
 * color range, then one row per variant. The colorbar is per *column* rather
 * than per panel because the range is shared — a bar on every panel would be
 * the same bar repeated, and would suggest each panel had its own scaling,
 * which is exactly the thing that would make the comparison a lie.
 */
async function buildGrid(
  opts: CompareOptions,
  sessions: ModelSession[],
  topo: SphereMeshTopology,
  showDt: boolean,
): Promise<{ rows: Row[]; rangeBars: { fill: (lo: number, hi: number) => void }[] }> {
  const { container, model } = opts;
  container.replaceChildren();
  container.classList.add('compare');

  const head = document.createElement('div');
  head.className = 'cmp-row cmp-head';
  const headSpacer = document.createElement('div');
  headSpacer.className = 'cmp-rowlabel';
  const headCols = document.createElement('div');
  headCols.className = 'cmp-cols';
  head.append(headSpacer, headCols);
  container.append(head);

  const rangeBars = model.species.map((name) => {
    const col = document.createElement('div');
    col.className = 'cmp-colhead';
    const tag = document.createElement('b');
    tag.textContent = name;
    const canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 8;
    canvas.className = 'cmp-rangebar';
    const lab = document.createElement('span');
    lab.className = 'cmp-rangelab';
    col.append(tag, canvas, lab);
    headCols.append(col);
    let painted = false;
    return {
      fill: (lo: number, hi: number): void => {
        const ctx = canvas.getContext('2d');
        if (ctx && !painted) {
          painted = true;
          const cmap = colormaps[opts.colormapName()] ?? colormaps.viridis;
          for (let x = 0; x < canvas.width; x++) {
            const [r, g, b] = cmap(x / (canvas.width - 1));
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(x, 0, 1, canvas.height);
          }
        }
        lab.textContent = `${fmtValue(lo)} … ${fmtValue(hi)}`;
      },
    };
  });

  const sphereBg = getComputedStyle(document.documentElement)
    .getPropertyValue('--sphere-bg')
    .trim();

  const rows: Row[] = [];
  for (let i = 0; i < sessions.length; i++) {
    const session = sessions[i];
    const variant = opts.variants[i];
    const color = VARIANT_COLORS[i % VARIANT_COLORS.length];

    const coords = await session.renderPositions();
    const posBuf = new Float32Array(topo.numVertices * 3);
    fillPositions(posBuf, coords, topo, opts.morph);

    const rowEl = document.createElement('div');
    rowEl.className = 'cmp-row';
    const labelEl = document.createElement('div');
    labelEl.className = 'cmp-rowlabel';
    labelEl.style.setProperty('--c', color);
    const nameEl = document.createElement('div');
    nameEl.className = 'cmp-rowname';
    nameEl.textContent = variantLabel(variant, showDt);
    const statEl = document.createElement('div');
    statEl.className = 'cmp-rowstat';
    labelEl.append(nameEl, statEl);
    const colsEl = document.createElement('div');
    colsEl.className = 'cmp-cols';
    rowEl.append(labelEl, colsEl);
    container.append(rowEl);

    const scenes: SphereScene[] = [];
    const valueBufs: Float32Array[] = [];
    const colorBufs: Float32Array[] = [];
    for (let k = 0; k < model.species.length; k++) {
      const box = document.createElement('div');
      box.className = 'sphere-box cmp-box';
      colsEl.append(box);
      const scene = new SphereScene(
        box,
        topo.numVertices,
        topo.indices,
        Float32Array.from(posBuf),
        sphereBg || undefined,
      );
      scene.fitCamera();
      scenes.push(scene);
      valueBufs.push(new Float32Array(topo.numVertices));
      colorBufs.push(new Float32Array(topo.numVertices * 3));
    }

    rows.push({
      variant, session, color, coords, posBuf, scenes, valueBufs, colorBufs,
      fields: [], err: model.species.map(() => 0), healthy: true, statEl,
    });
  }

  // Every panel shares one camera: the study is about the fields, and looking
  // at two of them from different angles is not comparing them.
  const all = rows.flatMap((r) => r.scenes);
  for (let i = 1; i < all.length; i++) all[0].syncCamerasWith(all[i]);

  return { rows, rangeBars };
}
