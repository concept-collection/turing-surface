import { requestShtDevice, describeAdapter } from './sht/sht.ts';
import { gridForLmax } from './sht/layout.ts';
import { ModelSession } from './mgpu/session.ts';
import { mModelByKey, presets, type MModel, type Params } from './mgpu/registry.ts';
import { ModelCompileError, formatFailure } from './mgpu/errors.ts';
import { EXTERNAL_OPS } from './mgpu/externals.ts';
import { CodeEditor } from './editor/codeEditor.ts';
import {
  formatCommand,
  resolvePreset,
  DEFAULT_NITER,
  DEFAULT_STEPS,
  DEFAULT_WARMUP,
  type RunSpec,
} from './bench/runSpec.ts';
import {
  mGeometries,
  mGeometryByKey,
  defaultGeometryParams,
  SPHERE_KEY,
  DEFAULT_GEOMETRY_KEY,
  type MGeometry,
} from './geom/registry.ts';
import {
  buildTopology,
  fillFieldValues,
  fillPositions,
  fillColors,
  type SphereMeshTopology,
} from './render/sphereMesh.ts';
import { SphereScene } from './render/SphereScene.ts';
import { Colorbar, fmtValue } from './render/colorbar.ts';
import { colormaps, colormapNames } from './render/colormaps.ts';
import { MovieRecorder } from './render/movie.ts';
import { CompareRun } from './compare/compareRun.ts';
import {
  crossProduct,
  mostResolved,
  variantKey,
  variantLabel,
  type Variant,
} from './compare/variants.ts';

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const elModel = $<HTMLSelectElement>('model');
const elGeometry = $<HTMLSelectElement>('geometry');
const elMorph = $<HTMLInputElement>('morph');
const elNiter = $<HTMLSelectElement>('niter');
const elLmax = $<HTMLSelectElement>('lmax');
const elOversample = $<HTMLSelectElement>('oversample');
const elColormap = $<HTMLSelectElement>('colormap');
const elRunPause = $<HTMLButtonElement>('runpause');
const elBenchmark = $<HTMLButtonElement>('benchmark');
const elReseed = $<HTMLButtonElement>('reseed');
const elResetView = $<HTMLButtonElement>('resetview');
const elMovieToggle = $<HTMLButtonElement>('movietoggle');
const elMovieBar = $('moviebar');
const elMovieSpeed = $<HTMLSelectElement>('moviespeed');
const elMovieRes = $<HTMLSelectElement>('movieres');
const elMovieRotate = $<HTMLInputElement>('movierotate');
const elMovie = $<HTMLButtonElement>('movie');
const elCompareToggle = $<HTMLButtonElement>('comparetoggle');
const elCompareBar = $('comparebar');
const elCmpNiter = $('cmp-niter');
const elCmpLmax = $('cmp-lmax');
const elCmpDt = $('cmp-dt');
const elCmpRef = $<HTMLSelectElement>('cmp-ref');
const elCmpStart = $<HTMLButtonElement>('cmp-start');
const elCmpCount = $('cmp-count');
const elParams = $('params');
const elGeomParams = $('geomparams');
const elGeomNote = $('geomnote');
const elPanels = $('panels');
const elStats = $('stats');
const elBenchResult = $('benchresult');
const elCmd = $('cmd');
const elCopyCmd = $<HTMLButtonElement>('copycmd');
const elBlurb = $('blurb');
const elErr = $('err');
const elSource = $<HTMLTextAreaElement>('source');
const elHighlight = $('highlight');
const elCompiled = $('compiled');
const elEditorTitle = $('editor-title');
const elEditorFile = $<HTMLSelectElement>('editor-file');
const elRecompile = $<HTMLButtonElement>('recompile');
const elRevert = $<HTMLButtonElement>('revert');

for (const p of presets) {
  const o = document.createElement('option');
  o.value = p.key;
  o.textContent = p.label;
  elModel.append(o);
}
for (const g of mGeometries) {
  const o = document.createElement('option');
  o.value = g.key;
  o.textContent = g.label;
  elGeometry.append(o);
}
for (const [value, label] of [['model', 'the solver'], ['geometry', 'the surface']]) {
  const o = document.createElement('option');
  o.value = value;
  o.textContent = label;
  elEditorFile.append(o);
}
for (const name of colormapNames) {
  const o = document.createElement('option');
  o.value = name;
  o.textContent = name;
  elColormap.append(o);
}
elColormap.value = 'jet';

/** Whichever .m is open: the solver or the surface. Both are MATLAB, compiled
 *  by the same backend, so one editor serves both. The host-provided operations
 *  are marked so the boundary between the file and what it is given is
 *  visible. */
const editor = new CodeEditor({
  textarea: elSource,
  overlay: elHighlight,
  external: EXTERNAL_OPS,
  onInput: (value) => {
    if (editing === 'geometry') editedGeomSource = value;
    else editedSource = value;
    elRecompile.textContent = 'Recompile *';
  },
});

/**
 * Timesteps submitted per rendered frame, at most. Nothing is read back
 * between them, so the batch costs one submit and one readback regardless of
 * size — but a compute pass is still real GPU work, and a browser's GPU
 * process enforces a watchdog timeout a headless desktop run does not: a
 * submission with enough dispatches in it can trip "device lost" outright,
 * on weak-enough hardware, well before it would ever show up as merely slow.
 * The `for k = 1:niter` correction loop makes a step's dispatch count scale
 * with niter (each iteration is ~15 dispatches per species — see
 * models/schnakenberg.m), so a fixed per-frame step count that was safe when
 * every model's step was a handful of dispatches is not safe once niter is
 * large. `stepsPerFrame`/`measureBurst` below scale it down — never up, so
 * the common case does not change — to keep one submission's total dispatch
 * count under DISPATCH_BUDGET regardless of how expensive the compiled step
 * is.
 */
const STEPS_PER_FRAME_BASE = 4;
/** See STEPS_PER_FRAME_BASE. Recomputed per rebuild in `rebuild()`. */
let stepsPerFrame = STEPS_PER_FRAME_BASE;

/**
 * Steps in a solver-timing burst, and how often to run one.
 *
 * Timing the solver needs a `queue.onSubmittedWorkDone()` to know the work
 * finished, and in a browser that is an IPC round trip into the GPU process — a
 * fixed cost of a few milliseconds. Spread over one frame's four steps it would
 * swamp them on a fast GPU and make the solver look far slower than it is. So the
 * rate is measured in an occasional larger batch, where the single sync is
 * amortized the way the desktop benchmark amortizes its own. The state is
 * snapshotted and restored around the batch, so measuring never advances the
 * simulation — otherwise the pattern would visibly lurch forward at every
 * measurement.
 */
const MEASURE_BURST_BASE = 32;
/** See STEPS_PER_FRAME_BASE — the measurement burst is one submission too,
 *  and a bigger one: 32 steps is the single largest batch this app ever
 *  submits, so it is the first thing to cross DISPATCH_BUDGET as niter grows. */
let measureBurst = MEASURE_BURST_BASE;
const MEASURE_EVERY_MS = 2000;

/**
 * Upper bound on dispatches in one submission — the frame batch and the
 * measurement burst are both scaled down to stay under this, never up, so
 * a cheap model's pacing is unchanged. Chosen well under what this project's
 * own desktop benchmark measures as trivially fast (single-digit ms even at
 * niter=8's ~450 dispatches/step), because the risk here is not GPU time on
 * capable hardware — it is a browser's GPU-process watchdog on weak
 * (integrated-graphics) hardware, which a headless desktop run never
 * exercises and this project has no way to benchmark directly.
 */
const DISPATCH_BUDGET = 1000;

/**
 * 'auto' display oversampling targets this many render latitudes: the factor is
 * the smallest power of two (up to 4) that reaches it. A solver grid already
 * this fine gains nothing visually and is not oversampled.
 */
const AUTO_RENDER_NLAT = 256;

/** The display oversampling factor the UI currently asks for. */
function resolveOversample(): number {
  if (elOversample.value !== 'auto') return Number(elOversample.value);
  const { nlat } = gridForLmax(Number(elLmax.value), model.pdeg);
  let os = 1;
  while (os < 4 && os * nlat < AUTO_RENDER_NLAT) os *= 2;
  return os;
}

/**
 * Movie frame rate, and a cap on frames per movie. Playback speed comes from
 * the UI, in simulation-time units per second of video; the movie's length is
 * the run's t at that speed, and the frame count follows from it — capped by
 * the run's own step count (a step is at most one frame) and by
 * MOVIE_MAX_FRAMES to bound encode time and file size. Frame timestamps are
 * derived from simulation time, so a capped movie keeps its duration and
 * speed exactly, at a lower effective frame rate.
 */
const MOVIE_FPS = 30;
const MOVIE_MAX_FRAMES = 3600;

/** Movie auto-rotation: camera revolutions per second of video. Measured in
 *  video time, so the orbit pace on screen is the same at every export speed. */
const MOVIE_ROTATE_RPS = 1 / 120;

// ---------------------------------------------------------------- state
let device: GPUDevice | null = null;
let session: ModelSession | null = null;
let topo: SphereMeshTopology | null = null;
let scenes: SphereScene[] = [];
let colorbars: Colorbar[] = [];
let valueBufs: Float32Array[] = [];
let colorBufs: Float32Array[] = [];
let ranges: { lo: number; hi: number }[] = [];
let resizeObs: ResizeObserver | null = null;

const initial = resolvePreset(presets[0].key);
let model: MModel = mModelByKey(initial.model.key)!;
let params: Params = initial.params;
let geometry: MGeometry = mGeometryByKey(DEFAULT_GEOMETRY_KEY)!;
let geomParams: Params = defaultGeometryParams(geometry);
/** Which file the editor is showing. */
let editing: 'model' | 'geometry' = 'model';
/** Each .m as edited in the page; `null` while it matches the file. */
let editedSource: string | null = null;
let editedGeomSource: string | null = null;
/** Sphere (0) to surface (1). Display only; does not touch the solver. */
let morph = 1;
let seed = 1;
let running = false;
let adapterName = '';
let pumping = false;
let movieBusy = false;
let movieCancel = false;
let solverMs = 0;
let frameMs = 0;
let lastMeasure = 0;
let generation = 0; // bumped on every rebuild to cancel stale pumps
/** Surface coordinates on the render grid, interleaved xyz; null before the
 *  first build. Kept so the morph slider can re-fill positions without
 *  re-synthesizing. */
let coords: Float32Array | null = null;
let posBuf: Float32Array | null = null;
/** The convergence study, when one is running; null in ordinary single-run
 *  mode. While it is non-null there is no `session`: the study owns one per
 *  variant, and the panels area is its grid. */
let compareRun: CompareRun | null = null;

const source = (): string => editedSource ?? model.source;
const geomSource = (): string => editedGeomSource ?? geometry.source;

// ---------------------------------------------------------------- UI wiring
function buildParamInputs(): void {
  elParams.replaceChildren();
  for (const spec of model.params) {
    const label = document.createElement('label');
    label.textContent = `${spec.label} `;
    const input = document.createElement('input');
    input.type = 'number';
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    input.value = String(params[spec.key]);
    input.addEventListener('change', () => {
      const v = Number(input.value);
      if (Number.isFinite(v)) params[spec.key] = v;
      // Parameters are uniforms, not constants baked into the kernels, so a
      // change costs an upload rather than a recompile. In compare mode `dt`
      // is the *base* timestep each variant's divisor divides, so the study
      // re-derives every variant's dt from it.
      session?.setParams(params);
      compareRun?.setParams(params);
      updateCommand();
    });
    label.append(input);
    elParams.append(label);
  }
}

/**
 * The shape's own parameters. Unlike the model's, these are NOT uniforms: the
 * surface is evaluated once at build time and reduced to coefficients, so
 * moving one rebuilds the geometry (and with it the mesh), though not the
 * simulation's compiled step.
 */
function buildGeomParamInputs(): void {
  elGeomParams.replaceChildren();
  if (geometry.params.length === 0) return;
  const tag = document.createElement('label');
  tag.textContent = `${geometry.key}.m`;
  elGeomParams.append(tag);
  for (const spec of geometry.params) {
    const label = document.createElement('label');
    label.textContent = `${spec.label} `;
    const input = document.createElement('input');
    input.type = 'number';
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    input.value = String(geomParams[spec.key]);
    input.addEventListener('change', () => {
      const v = Number(input.value);
      if (Number.isFinite(v)) geomParams[spec.key] = v;
      viewChange = viewChange.then(() => applyGeometry());
    });
    label.append(input);
    elGeomParams.append(label);
  }
}

function applyPreset(presetKey: string): void {
  const resolved = resolvePreset(presetKey);
  const next = mModelByKey(resolved.model.key);
  if (!next) {
    elErr.textContent = `No .m model for '${resolved.model.key}'`;
    return;
  }
  model = next;
  params = resolved.params;
  editedSource = null;
  buildParamInputs();
  elBlurb.textContent = model.blurb;
  showEditorFile();
  updateCommand();
}

function applyGeometryChoice(key: string): void {
  const next = mGeometryByKey(key);
  if (!next) {
    elErr.textContent = `No .m geometry for '${key}'`;
    return;
  }
  geometry = next;
  geomParams = defaultGeometryParams(geometry);
  editedGeomSource = null;
  buildGeomParamInputs();
  showEditorFile();
}

/** Load the chosen file into the editor, keeping any unsaved edit to it. */
function showEditorFile(): void {
  editing = elEditorFile.value === 'geometry' ? 'geometry' : 'model';
  if (editing === 'geometry') {
    editor.value = geomSource();
    elEditorTitle.textContent = `geometries/${geometry.key}.m`;
  } else {
    editor.value = source();
    elEditorTitle.textContent = `models/${model.key}.m`;
  }
}

/**
 * The run currently on screen, as the benchmark's RunSpec. While a study is
 * running there is no single run, so this describes its *reference* variant —
 * the one the other rows are measured against, and the only one of them whose
 * numbers mean anything on their own.
 */
function currentSpec(): RunSpec {
  const ref = compareRun?.variants[compareRefIndex()];
  const dt = ref ? { dt: (params.dt ?? 0) / ref.dtDiv } : null;
  return {
    preset: elModel.value,
    lmax: ref ? ref.lmax : Number(elLmax.value),
    seed,
    steps: DEFAULT_STEPS,
    warmup: DEFAULT_WARMUP,
    params: dt ? { ...params, ...dt } : params,
    geometry: geometry.key,
    geometryParams: geomParams,
    niter: ref ? ref.niter : Number(elNiter.value),
  };
}

function updateCommand(): void {
  elCmd.textContent = formatCommand(currentSpec());
}

elModel.addEventListener('change', () => {
  applyPreset(elModel.value);
  void rebuild();
});
elLmax.addEventListener('change', () => void rebuild());
// The solve iteration count is unrolled into the compiled step, so unlike a
// parameter it cannot be changed without recompiling.
elNiter.addEventListener('change', () => void rebuild());
// Oversampling and geometry are display-or-data changes, not code ones, so
// they swap things in place rather than rebuilding the run. Serialized through
// one chain: a rapid second change waits its turn.
let viewChange = Promise.resolve();
elOversample.addEventListener('change', () => {
  // The study picks its own display grid — one grid common to every variant is
  // what makes their fields comparable — so this control is inert (and
  // disabled) while one is running.
  if (compareRun) return;
  viewChange = viewChange.then(() => applyOversample());
});
elGeometry.addEventListener('change', () => {
  applyGeometryChoice(elGeometry.value);
  viewChange = viewChange.then(() => applyGeometry());
});
// Morph is pure rendering: no readback, no GPU work, just the vertex buffer.
elMorph.addEventListener('input', () => {
  morph = Number(elMorph.value);
  if (compareRun) compareRun.setMorph(morph);
  else applyMorph();
});
elColormap.addEventListener('change', () => {
  if (compareRun) void compareRun.draw();
  else void draw();
});
elEditorFile.addEventListener('change', () => showEditorFile());

function setRunning(next: boolean): void {
  running = next;
  elRunPause.textContent = running ? 'Pause' : 'Run';
  if (compareRun) {
    compareRun.setRunning(next);
    return;
  }
  if (running) void pump();
}

elRunPause.addEventListener('click', () => setRunning(!running));
elBenchmark.addEventListener('click', () => void benchmark());
elReseed.addEventListener('click', () => {
  seed = (Math.random() * 2 ** 31) >>> 0;
  setRunning(false);
  updateCommand();
  void reseed();
});
elResetView.addEventListener('click', () => {
  compareRun?.resetView();
  for (const s of scenes) s.resetCamera();
});
elMovieToggle.addEventListener('click', () => {
  elMovieBar.hidden = !elMovieBar.hidden;
});
elMovie.addEventListener('click', () => {
  if (movieBusy) movieCancel = true;
  else void recordMovie();
});

elRecompile.addEventListener('click', () => {
  if (editing === 'geometry') editedGeomSource = editor.value;
  else editedSource = editor.value;
  void rebuild();
});
elRevert.addEventListener('click', () => {
  if (editing === 'geometry') editedGeomSource = null;
  else editedSource = null;
  showEditorFile();
  void rebuild();
});

// The command reproduces this run's parameters on the desktop; keep it
// selectable even where the clipboard API is unavailable.
elCopyCmd.addEventListener('click', () => {
  const text = elCmd.textContent ?? '';
  const flash = (msg: string): void => {
    elCopyCmd.textContent = msg;
    setTimeout(() => (elCopyCmd.textContent = 'Copy'), 1200);
  };
  const selectCommand = (): void => {
    const range = document.createRange();
    range.selectNodeContents(elCmd);
    const sel = getSelection();
    sel?.removeAllRanges();
    sel?.addRange(range);
    flash('Selected');
  };
  if (!navigator.clipboard) return selectCommand();
  navigator.clipboard.writeText(text).then(() => flash('Copied'), selectCommand);
});

// ---------------------------------------------------------------- setup
function disposeView(): void {
  for (const s of scenes) s.dispose();
  scenes = [];
  colorbars = [];
  topo = null;
  coords = null;
  posBuf = null;
  resizeObs?.disconnect();
  resizeObs = null;
  elPanels.replaceChildren();
}

/**
 * Build the mesh, scenes, colorbars and per-species buffers on the current
 * render grid, from surface coordinates already synthesized there. Call
 * disposeView() first. The color ranges are kept if present, so a display-only
 * rebuild (an oversampling change) does not pop the shading; a full rebuild
 * clears `ranges` beforehand.
 */
function buildView(surface: Float32Array): void {
  if (!session) return;
  const view = session.viewSht;
  const { nphi } = view.cfg;
  const phi = new Float64Array(nphi);
  for (let j = 0; j < nphi; j++) phi[j] = (2 * Math.PI * j) / nphi;
  topo = buildTopology(view.cosTheta, phi);
  coords = surface;
  posBuf = new Float32Array(topo.numVertices * 3);
  fillPositions(posBuf, coords, topo, morph);

  const sphereBg = getComputedStyle(document.documentElement)
    .getPropertyValue('--sphere-bg')
    .trim();
  for (let k = 0; k < model.species.length; k++) {
    const panel = document.createElement('div');
    panel.className = 'panel';
    const box = document.createElement('div');
    box.className = 'sphere-box';
    const tag = document.createElement('div');
    tag.className = 'species-tag';
    tag.textContent = model.species[k];
    box.append(tag);
    const side = document.createElement('div');
    panel.append(box, side);
    elPanels.append(panel);

    const scene = new SphereScene(
      box,
      topo.numVertices,
      topo.indices,
      // Each scene owns its position buffer: three.js uploads from it, and the
      // morph rewrites all of them from the one shared `coords`.
      Float32Array.from(posBuf),
      sphereBg || undefined,
    );
    scene.fitCamera();
    scenes.push(scene);
    colorbars.push(new Colorbar(side));
    valueBufs[k] = new Float32Array(topo.numVertices);
    colorBufs[k] = new Float32Array(topo.numVertices * 3);
    if (!ranges[k]) ranges[k] = { lo: NaN, hi: NaN };
  }
  for (let k = 1; k < scenes.length; k++) scenes[0].syncCamerasWith(scenes[k]);

  resizeObs = new ResizeObserver(() => {
    const boxes = elPanels.querySelectorAll<HTMLElement>('.sphere-box');
    boxes.forEach((box, i) => {
      scenes[i]?.resize(box.clientWidth, box.clientHeight);
    });
  });
  elPanels
    .querySelectorAll<HTMLElement>('.sphere-box')
    .forEach((box) => resizeObs!.observe(box));
}

/**
 * Apply the UI's oversampling choice to the running session. Display-only: the
 * session and its state survive; only the display plan, mesh and scenes are
 * rebuilt, keeping the camera pose and color ranges. The pump is drained first
 * so no readback is in flight on the plan being replaced.
 */
async function applyOversample(): Promise<void> {
  if (!session) return;
  const gen = generation;
  const os = resolveOversample();
  if (os === session.oversample) return;
  const wasRunning = running;
  setRunning(false);
  while (pumping) await nextFrame();
  if (gen !== generation || !session) return;
  await session.setOversample(os);
  if (gen !== generation || !session) return;
  const surface = await session.renderPositions();
  if (gen !== generation || !session) return;
  const cam = scenes[0]?.cameraState();
  disposeView();
  buildView(surface);
  if (cam) for (const s of scenes) s.setCameraState(cam);
  await draw();
  updateStats();
  if (wasRunning) setRunning(true);
}

/**
 * Re-evaluate the surface and swap it in. Data, not code: the compiled step is
 * untouched and the simulation keeps its state and its model time, so a shape
 * can be changed mid-run. Only the mesh is rebuilt.
 */
async function applyGeometry(): Promise<void> {
  // The in-place swap below is a single session's trick. Each variant carries
  // the surface band-limited at its own lmax, and the study's meshes are built
  // from those, so a shape change goes through the full rebuild instead.
  if (compareRun) return rebuildCompare();
  if (!session) return;
  const gen = generation;
  const wasRunning = running;
  setRunning(false);
  while (pumping) await nextFrame();
  if (gen !== generation || !session) return;
  try {
    await session.setGeometry(geometry, geomParams, geomSource());
  } catch (e) {
    reportCompileError(e);
    return;
  }
  if (gen !== generation || !session) return;
  const surface = await session.renderPositions();
  if (gen !== generation || !session) return;
  const cam = scenes[0]?.cameraState();
  disposeView();
  buildView(surface);
  if (cam) for (const s of scenes) s.setCameraState(cam);
  elErr.textContent = '';
  await draw();
  updateGeomNote();
  updateStats();
  if (wasRunning) setRunning(true);
}

/** Re-place the vertices for the current morph. No GPU work and no readback —
 *  the surface is already on the CPU, so this is a buffer fill per panel. */
function applyMorph(): void {
  if (!topo || !coords || !posBuf) return;
  fillPositions(posBuf, coords, topo, morph);
  for (const s of scenes) s.updatePositions(posBuf);
}

/** What the surface is, and the standing caveat about where it is not. */
function updateGeomNote(): void {
  // In compare mode each variant carries the surface band-limited at its own
  // lmax; the reference's is the one quoted, as everywhere else.
  const s = session ?? compareRun?.referenceSession ?? null;
  if (!s) {
    elGeomNote.textContent = '';
    return;
  }
  const { lo, hi } = s.geometry.radiusRange();
  const isSphere = s.geometryModel.key === SPHERE_KEY;
  elGeomNote.innerHTML =
    `<b>${s.geometryModel.label}</b> — ${s.geometryModel.blurb} ` +
    `Radius ${lo.toFixed(3)}–${hi.toFixed(3)}.` +
    (isSphere ? '' : ' <b>Rendered only</b> — not yet in the operator.');
}

/** Report a compile failure, and select the offending text in the editor. */
function reportCompileError(e: unknown): void {
  elErr.textContent = formatFailure(e, source());
  elCompiled.textContent = '';
  if (e instanceof ModelCompileError && e.start !== undefined) {
    editor.select(e.start, e.end ?? e.start);
  }
}

async function rebuild(): Promise<void> {
  // A study is several runs, so "rebuild the run" means rebuild all of them.
  // Everything that recompiles — a model or preset change, an edit to either
  // .m, a revert — arrives here, and none of it needs to know which mode is up.
  if (compareRun) return rebuildCompare();
  generation++;
  const gen = generation;
  setRunning(false);
  disposeView();
  session?.destroy();
  session = null;
  solverMs = 0;
  frameMs = 0;
  // Not 0: with a large niter's dispatch count not yet known (that needs the
  // compiled plan below), the first measurement burst should wait for the
  // ordinary per-frame batch — already sized to this model — to prove itself
  // first, rather than firing a possibly-oversized burst before a single
  // frame has run.
  lastMeasure = performance.now();
  elErr.textContent = '';
  updateCommand();
  if (!device) return;

  try {
    session = await ModelSession.create({
      device,
      model,
      params,
      lmax: Number(elLmax.value),
      source: source(),
      oversample: resolveOversample(),
      geometry,
      geometryParams: geomParams,
      geometrySource: geomSource(),
      niter: Number(elNiter.value),
    });
  } catch (e) {
    reportCompileError(e);
    return;
  }
  if (gen !== generation) return;

  session.seed(seed);

  const plan = session.describe();
  elCompiled.textContent =
    `one step compiled to ${plan.step.length} GPU operations:\n` +
    plan.step.map((l) => `  ${l}`).join('\n');
  elRecompile.textContent = 'Recompile';

  // Scale the frame batch and the measurement burst down — never up — so
  // neither submission's total dispatch count exceeds DISPATCH_BUDGET, no
  // matter how expensive niter has made one step. See STEPS_PER_FRAME_BASE.
  const opsPerStep = Math.max(1, plan.step.length);
  stepsPerFrame = Math.max(1, Math.min(STEPS_PER_FRAME_BASE, Math.floor(DISPATCH_BUDGET / opsPerStep)));
  measureBurst = Math.max(1, Math.min(MEASURE_BURST_BASE, Math.floor(DISPATCH_BUDGET / opsPerStep)));

  const surface = await session.renderPositions();
  if (gen !== generation) return;

  ranges = [];
  buildView(surface);

  await draw();
  updateGeomNote();
  updateStats();
  void pump();
}

async function reseed(): Promise<void> {
  // One new perturbation for the whole study, band-limited at its coarsest
  // variant and evaluated on each grid — see src/compare/sharedStart.ts.
  if (compareRun) return compareRun.reseed(seed);
  if (!session) return;
  const gen = generation;
  session.seed(seed);
  if (gen !== generation) return;
  for (const r of ranges) {
    r.lo = NaN;
    r.hi = NaN;
  }
  await draw();
  updateStats();
}

// ---------------------------------------------------------------- drawing
async function draw(): Promise<void> {
  if (!session || !topo) return;
  const gen = generation;
  const cmap = colormaps[elColormap.value] ?? colormaps.viridis;
  for (let k = 0; k < model.species.length; k++) {
    // The one readback per frame — the loop is otherwise entirely on the GPU.
    // A rebuild can land while this is in flight and destroy the buffer being
    // mapped, which rejects the map; that result is stale anyway, so drop it.
    let field: Float32Array;
    try {
      field = await session.readSpecies(k);
    } catch (e) {
      if (gen !== generation) return;
      throw e;
    }
    if (gen !== generation || !topo) return;
    fillFieldValues(valueBufs[k], field, topo);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of valueBufs[k]) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    // smooth the color range in both directions so the shading evolves
    // gently as the pattern grows (out-of-range values clamp meanwhile)
    const r = ranges[k];
    if (!Number.isFinite(r.lo)) {
      r.lo = lo;
      r.hi = hi;
    } else {
      const a = 0.15;
      r.lo += a * (lo - r.lo);
      r.hi += a * (hi - r.hi);
    }
    if (r.hi - r.lo < 1e-9) {
      const mid = (r.hi + r.lo) / 2;
      r.lo = mid - 5e-10;
      r.hi = mid + 5e-10;
    }
    fillColors(colorBufs[k], valueBufs[k], r.lo, r.hi, cmap);
    scenes[k]?.updateColors(colorBufs[k]);
    colorbars[k]?.update(cmap, r.lo, r.hi);
  }
}

function updateStats(): void {
  if (!session) return;
  const { nlat, nphi } = session.cfg;
  const kind = `WebGPU fp32${adapterName ? ` — ${adapterName}` : ''}`;
  const solver =
    solverMs > 0
      ? `<b>${solverMs.toFixed(2)} ms/step</b> (${(1000 / solverMs).toFixed(0)} steps/s)`
      : '—';
  const frame = frameMs > 0 ? `${frameMs.toFixed(1)} ms/frame` : '—';
  const view = session.viewSht.cfg;
  const render =
    session.oversample > 1
      ? ` (display ${view.nlat}×${view.nphi})`
      : '';
  elStats.innerHTML =
    `<b>${kind}</b> · grid ${nlat}×${nphi}${render} · nlm ${session.sht.nlm.toLocaleString()} · ` +
    `solver ${solver} · ${frame} · ` +
    `t = <b>${session.t.toFixed(2)}</b> (${session.steps} steps)`;
}

// ---------------------------------------------------------------- sim loop
const nextFrame = () => new Promise<number>(requestAnimationFrame);

async function pump(): Promise<void> {
  if (pumping) return;
  pumping = true;
  const gen = generation;
  try {
    while (running && session && gen === generation) {
      // Occasionally, a burst purely to measure the solver rate: many steps,
      // one sync, nothing read back — directly comparable to the desktop
      // benchmark's throughput number. State-preserving: the display and
      // model time are unaffected.
      if (performance.now() - lastMeasure > MEASURE_EVERY_MS) {
        const ms = await session.measure(measureBurst);
        if (gen !== generation) break;
        solverMs = ms;
        lastMeasure = performance.now();
      }

      // The frame itself. No explicit sync here — draw()'s readback already
      // waits for the steps, so asking twice would only add a round trip.
      const t0 = performance.now();
      session.step(stepsPerFrame);
      await draw();
      if (gen !== generation) break;
      frameMs = frameMs === 0
        ? performance.now() - t0
        : frameMs + 0.05 * (performance.now() - t0 - frameMs);
      updateStats();
      await nextFrame();
    }
    if (gen === generation) {
      await draw();
      updateStats();
    }
  } finally {
    pumping = false;
  }
}

/**
 * Sustained solver benchmark, in the page.
 *
 * The same measurement `npm run bench` makes: batches of steps submitted
 * together, waited for, never read back, with no rendering and no animation
 * pacing in between. That makes it directly comparable to the terminal number,
 * which is the only way to tell a genuinely slower browser GPU stack apart from
 * the costs the app adds on top.
 *
 * It also reports the ramp — the first third of the run against the last. GPUs
 * downclock when idle, and an animation-paced loop leaves them idle most of every
 * frame, so a large ramp means the app's steady-state number is limited by clocks
 * rather than by the work.
 *
 * These are ordinary steps: the simulation advances by them.
 */
async function benchmark(): Promise<void> {
  if (!session || movieBusy) return;
  setRunning(false);
  // Same base size and the same DISPATCH_BUDGET scaling as the automatic
  // measurement burst (see STEPS_PER_FRAME_BASE) — this is a user-triggered
  // 32-step submission, exactly the shape of thing that risks a browser's
  // GPU-process watchdog on weak hardware once niter makes a step expensive.
  const BATCH = measureBurst;
  const DURATION_MS = 2000;
  elBenchResult.textContent = 'benchmarking…';
  // A movie started mid-benchmark would replay while this loop still steps.
  elMovie.disabled = true;
  try {
    await nextFrame();

    const gen = generation;
    const perStep: number[] = [];
    const t0 = performance.now();
    while (performance.now() - t0 < DURATION_MS) {
      const b0 = performance.now();
      session.step(BATCH);
      await session.sync();
      if (gen !== generation) return;
      perStep.push((performance.now() - b0) / BATCH);
    }

    const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
    const all = mean(perStep);
    const best = Math.min(...perStep);
    const third = Math.max(1, Math.floor(perStep.length / 3));
    const first = mean(perStep.slice(0, third));
    const last = mean(perStep.slice(-third));
    const steps = perStep.length * BATCH;

    elBenchResult.innerHTML =
      `sustained solver: <b>${all.toFixed(2)} ms/step</b> ` +
      `(${(1000 / all).toFixed(0)} steps/s) · best ${best.toFixed(2)} · ` +
      `ramp ${(first / last).toFixed(2)}× · ${steps} steps · ` +
      `compare with <code>npm run bench -- --lmax ${session.cfg.lmax}</code>`;
    await draw();
    updateStats();
  } finally {
    elMovie.disabled = false;
  }
}

// ---------------------------------------------------------------- movie
function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** Submit `n` steps in bounded command buffers — a single buffer encoding
 *  many thousands of steps can exhaust the encoder. */
function submitSteps(n: number): void {
  while (n > 0 && session) {
    const chunk = Math.min(512, n);
    session.step(chunk);
    n -= chunk;
  }
}

/** While recording, lock everything that could change the run mid-replay;
 *  the Movie button itself becomes the cancel button. */
function setMovieUi(on: boolean): void {
  const locked = [
    elModel, elGeometry, elMorph, elNiter, elLmax, elOversample, elColormap,
    elRunPause, elBenchmark, elReseed, elRecompile, elRevert, elEditorFile,
    elMovieSpeed, elMovieRes, elMovieRotate, elMovieToggle,
  ];
  for (const el of locked) el.disabled = on;
  elParams.querySelectorAll('input').forEach((input) => (input.disabled = on));
  elGeomParams.querySelectorAll('input').forEach((input) => (input.disabled = on));
  elMovie.textContent = on ? 'Cancel · 0%' : 'Export';
}

/**
 * Recompute the run from t = 0 and download it as an MP4.
 *
 * The movie is not a recording of what already happened — it is the same
 * trajectory recomputed: same seed, same source, and the *current* parameters
 * and colormap throughout. Determinism makes this exact: after the replay the
 * state is where it was, so the one session is reused and the app resumes as
 * if nothing happened. Frames are composited from the live panels, so the
 * movie shows the spheres at the current camera orientation — and the replay
 * doubles as the progress display, since it is visible on screen.
 */
async function recordMovie(): Promise<void> {
  if (!session || movieBusy) return;
  if (session.steps === 0) {
    elMovie.textContent = 'run first';
    setTimeout(() => (elMovie.textContent = 'Export'), 1200);
    return;
  }
  movieBusy = true;
  movieCancel = false;
  const gen = generation;
  setMovieUi(true);
  let wasRunning = false;
  let total = 0;
  let done = 0;
  let seeded = false;
  let camBefore: ReturnType<SphereScene['cameraState']> | undefined;
  try {
    // An in-flight display-grid swap replaces the scenes whose canvases the
    // recorder captures, and resumes the run when it lands — let it finish.
    await viewChange;
    if (gen !== generation || !session) return;
    wasRunning = running;
    setRunning(false);
    while (pumping) await nextFrame(); // let an in-flight live frame drain
    if (gen !== generation || !session) return;
    total = session.steps;
    const speed = Number(elMovieSpeed.value) || 10;
    const sphere = Number(elMovieRes.value) || 768;
    const rotate = elMovieRotate.checked;
    if (rotate) camBefore = scenes[0]?.cameraState();
    // Render the scenes at exactly the chosen resolution for the recording —
    // independent of the window size — and restore afterwards.
    for (const s of scenes) s.captureSize(sphere);
    const durationS = Math.max(session.t / speed, 2 / MOVIE_FPS);
    const frames = Math.max(
      2,
      Math.min(Math.round(durationS * MOVIE_FPS) + 1, total + 1, MOVIE_MAX_FRAMES),
    );
    /** The step index captured as frame `i`; both endpoints land exactly. */
    const stepAt = (i: number): number => Math.round((i * total) / (frames - 1));

    const title =
      (presets.find((p) => p.key === elModel.value)?.label ?? model.label) +
      ` on ${geometry.label.toLowerCase()}` +
      (editedSource !== null || editedGeomSource !== null ? ' (edited)' : '');
    const subtitle = model.params
      .map((spec) => `${spec.label} ${fmtValue(params[spec.key])}`)
      .join(' · ');
    const rec = await MovieRecorder.create({
      panels: model.species.map((label, k) => ({ canvas: scenes[k].canvas, label })),
      title,
      subtitle,
      speed,
      fps: (frames - 1) / durationS,
      sphere,
    });

    let finished = false;
    try {
      // Reset the color-range smoothing as a re-seed does, so the shading
      // evolves in the movie the way it did live.
      session.seed(seed);
      seeded = true;
      for (const r of ranges) {
        r.lo = NaN;
        r.hi = NaN;
      }
      const cmap = colormaps[elColormap.value] ?? colormaps.viridis;
      let lastVideoS = 0;
      for (let frame = 0; ; ) {
        await draw();
        if (gen !== generation) return;
        if (movieCancel) break;
        if (rotate) {
          // Advance the orbit by this frame's share of video time; siblings
          // follow scenes[0] through the usual camera sync.
          const videoS = session.t / speed;
          scenes[0]?.orbitBy(2 * Math.PI * MOVIE_ROTATE_RPS * (videoS - lastVideoS));
          lastVideoS = videoS;
        }
        for (const s of scenes) s.renderNow();
        await rec.addFrame(
          session.t,
          model.species.map((_, k) => ({ cmap, lo: ranges[k].lo, hi: ranges[k].hi })),
        );
        if (++frame >= frames) {
          finished = true;
          break;
        }
        const target = stepAt(frame);
        submitSteps(target - done);
        done = target;
        elMovie.textContent = `Cancel · ${Math.round((100 * done) / total)}%`;
      }
      if (finished) {
        const blob = await rec.finish();
        saveBlob(
          blob,
          `turing-surface-${model.key}-${geometry.key}-` +
            `t${session.t.toFixed(2)}-${speed}x.mp4`,
        );
      }
    } finally {
      if (!finished) rec.cancel();
    }
  } catch (e) {
    elErr.textContent = `movie: ${e instanceof Error ? e.message : e}`;
  } finally {
    // A cancelled replay stopped short of where the run was; step the
    // remainder — determinism makes this land exactly there.
    if (seeded && gen === generation && session) {
      while (done < total && gen === generation && session) {
        const n = Math.min(4096, total - done);
        submitSteps(n);
        done += n;
        elMovie.textContent = `restoring · ${Math.round((100 * done) / total)}%`;
        await session.sync();
      }
      await draw();
      updateStats();
    }
    if (gen === generation) {
      for (const s of scenes) s.restoreSize();
    }
    if (camBefore && gen === generation) {
      for (const s of scenes) s.setCameraState(camBefore);
    }
    movieBusy = false;
    setMovieUi(false);
    if (gen === generation) setRunning(wasRunning);
  }
}

// ---------------------------------------------------------------- compare
/**
 * Comparing several solver settings at once.
 *
 * Deliberately a mode rather than a widening of the ordinary controls: the
 * single-run path above is untouched, and with the bar closed nothing about
 * using this page has changed. Opening it and pressing Compare tears down the
 * one session and hands the panels area to a CompareRun, which owns a session
 * per variant; pressing it again puts the single run back.
 *
 * The ceilings below are not arbitrary. Each variant compiles its whole
 * unrolled step with no pipeline cache between sessions (a solve iteration is
 * ~15 kernels per species), so the variant count is what you wait for; and
 * each panel is a WebGL context and a full mesh, so the panel count is what
 * the browser has to keep alive at once.
 */
const MAX_VARIANTS = 6;
const MAX_PANELS = 12;
/** dt divisors. Powers of two so that dtBase/K is exact in binary and every
 *  variant lands on the same model time with no accumulated drift. */
const DT_DIVISORS = [1, 2, 4, 8];

/**
 * What the bar opens on: the default iteration count against the next step up,
 * at the default band. Two variants, so the first study is quick to compile,
 * and it asks the question the control exists for — is the default already
 * converged? A flat, low curve says yes; one that climbs says the answer is
 * still moving at niter 8 and the default is not enough for this shape.
 */
const cmpSelected = {
  niter: new Set<number>([DEFAULT_NITER, 2 * DEFAULT_NITER]),
  lmax: new Set<number>([63]),
  dt: new Set<number>([1]),
};

/** A row of toggle chips backed by a Set. At least one stays selected — an
 *  empty axis has no meaning here, and silently falling back to a default
 *  would hide which values are actually being run. */
function buildChips(host: HTMLElement, values: number[], selected: Set<number>, label: (v: number) => string): void {
  host.replaceChildren();
  for (const value of values) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.textContent = label(value);
    const paint = (): void => chip.setAttribute('aria-pressed', String(selected.has(value)));
    paint();
    chip.addEventListener('click', () => {
      if (selected.has(value)) {
        if (selected.size === 1) return;
        selected.delete(value);
      } else {
        selected.add(value);
      }
      paint();
      refreshVariants();
    });
    host.append(chip);
  }
}

const cmpVariants = (): Variant[] =>
  crossProduct([...cmpSelected.niter], [...cmpSelected.lmax], [...cmpSelected.dt]);

/** The reference the user picked, clamped to the current variant list. */
let cmpRefKey = '';

/** Index of the reference in the current variant list, never negative. */
function compareRefIndex(): number {
  const i = cmpVariants().map(variantKey).indexOf(cmpRefKey);
  return i < 0 ? 0 : i;
}

function refreshVariants(): void {
  const variants = cmpVariants();
  const showDt = cmpSelected.dt.size > 1;
  const panels = variants.length * model.species.length;

  const prev = cmpRefKey;
  elCmpRef.replaceChildren();
  for (const v of variants) {
    const o = document.createElement('option');
    o.value = variantKey(v);
    o.textContent = variantLabel(v, showDt);
    elCmpRef.append(o);
  }
  const keys = variants.map(variantKey);
  cmpRefKey = keys.includes(prev) ? prev : keys[mostResolved(variants)];
  elCmpRef.value = cmpRefKey;

  const tooMany =
    variants.length > MAX_VARIANTS
      ? `${variants.length} variants — at most ${MAX_VARIANTS}`
      : panels > MAX_PANELS
        ? `${panels} panels — at most ${MAX_PANELS}`
        : '';
  elCmpCount.textContent = tooMany
    ? `too many: ${tooMany}`
    : `${variants.length} variants × ${model.species.length} species = ${panels} panels`;
  elCmpCount.style.color = tooMany ? '#b35900' : '';
  elCmpStart.disabled = tooMany !== '' && compareRun === null;
}

buildChips(
  elCmpNiter,
  [...elNiter.options].map((o) => Number(o.value)),
  cmpSelected.niter,
  String,
);
buildChips(
  elCmpLmax,
  [...elLmax.options].map((o) => Number(o.value)),
  cmpSelected.lmax,
  String,
);
buildChips(elCmpDt, DT_DIVISORS, cmpSelected.dt, (v) => (v === 1 ? 'dt' : `dt/${v}`));
refreshVariants();

elCmpRef.addEventListener('change', () => {
  cmpRefKey = elCmpRef.value;
  if (compareRun) void rebuildCompare();
});

elCompareToggle.addEventListener('click', () => {
  elCompareBar.hidden = !elCompareBar.hidden;
});

elCmpStart.addEventListener('click', () => {
  if (compareRun) void stopCompare();
  else void startCompare();
});

/** Controls the study supersedes or cannot honour while it is running. */
function setCompareUi(on: boolean): void {
  for (const el of [elNiter, elLmax, elOversample, elBenchmark, elMovieToggle]) {
    el.disabled = on;
  }
  elCmpNiter.querySelectorAll('button').forEach((b) => (b.disabled = on));
  elCmpLmax.querySelectorAll('button').forEach((b) => (b.disabled = on));
  elCmpDt.querySelectorAll('button').forEach((b) => (b.disabled = on));
  elCmpStart.textContent = on ? 'Stop comparing' : 'Compare';
  elCompareToggle.textContent = on ? 'Comparing' : 'Compare';
  if (on) elMovieBar.hidden = true;
}

async function startCompare(): Promise<void> {
  if (compareRun || !device) return;
  const variants = cmpVariants();
  if (variants.length > MAX_VARIANTS || variants.length * model.species.length > MAX_PANELS) {
    return;
  }
  // Take down the single run first: its pump, its scenes, its session. The
  // generation bump makes any readback already in flight drop its result.
  generation++;
  setRunning(false);
  while (pumping) await nextFrame();
  disposeView();
  session?.destroy();
  session = null;
  elBenchResult.textContent = '';
  elErr.textContent = '';
  setCompareUi(true);

  try {
    compareRun = await CompareRun.create({
      device,
      model,
      params,
      source: source(),
      geometry,
      geometryParams: geomParams,
      geometrySource: geomSource(),
      variants,
      reference: compareRefIndex(),
      seed,
      morph,
      colormapName: () => elColormap.value,
      container: elPanels,
      onStatus: (html) => (elStats.innerHTML = html),
    });
  } catch (e) {
    compareRun = null;
    setCompareUi(false);
    refreshVariants();
    reportCompileError(e);
    await rebuild();
    return;
  }
  updateGeomNote();
  // The command describes the reference variant, which only exists now.
  updateCommand();
  elRunPause.textContent = 'Run';
}

async function stopCompare(): Promise<void> {
  if (!compareRun) return;
  compareRun.dispose();
  compareRun = null;
  setCompareUi(false);
  refreshVariants();
  elStats.textContent = '';
  await rebuild();
}

/** Rebuild the study in place — after a model, geometry, source or reference
 *  change. Same teardown as stopping, without leaving the mode. */
async function rebuildCompare(): Promise<void> {
  if (!compareRun) return;
  compareRun.dispose();
  compareRun = null;
  setCompareUi(false);
  await startCompare();
}

// ---------------------------------------------------------------- boot
async function boot(): Promise<void> {
  elModel.value = presets[0].key;
  // The iteration count is one default shared with the benchmark, like the
  // rest of the RunSpec's — take it from there rather than from the markup, so
  // the page and `npm run bench` cannot start out disagreeing about it.
  elNiter.value = String(DEFAULT_NITER);
  elGeometry.value = DEFAULT_GEOMETRY_KEY;
  elMorph.value = String(morph);
  applyGeometryChoice(DEFAULT_GEOMETRY_KEY);
  applyPreset(presets[0].key);
  try {
    device = await requestShtDevice();
    adapterName = await describeAdapter(device);
  } catch (e) {
    device = null;
    elErr.textContent =
      `WebGPU is not available (${e instanceof Error ? e.message : e}). ` +
      `Use a WebGPU-capable browser such as Chrome or Edge.`;
    return;
  }
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') {
      elErr.textContent = `WebGPU device lost: ${info.message}`;
    }
  });
  await rebuild();
}

void boot();
