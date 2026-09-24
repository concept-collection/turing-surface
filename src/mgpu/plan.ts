/**
 * Statement list -> a replayable sequence of GPU operations.
 *
 * Everything expensive happens once, here: pipeline compilation, buffer
 * allocation, bind-group construction. Because numbl fixes every type and
 * shape at lowering time, the resulting op sequence is fully static — so
 * `encodeStep` is pure synchronous command recording, with no allocation, no
 * pipeline lookup and no readback. That is what lets the whole timestep be
 * encoded into one submit and keeps the CPU out of the loop.
 */
import { isMultiElement, scalarDouble } from 'numbl-src/numbl-core/jit/lowering/types.ts';
import type {
  Assign,
  For,
  IRExpr,
  IRStmt,
  MultiAssignCall,
} from 'numbl-src/numbl-core/jit/lowering/ir.ts';
import type { NumericType, Type } from 'numbl-src/numbl-core/jit/lowering/types.ts';
import { ShtPlan, type ShtBinding, type ShtBatchBinding, type ShtDphigBinding } from '../sht/sht.ts';
import { DerivPlan, type DerivBinding } from '../sht/deriv.ts';
import { ReducePlan, type DotBinding } from './reduce.ts';
import { exactMatrixBytes, type ExactOperator, type ExactSite } from './exact.ts';
import type { CompiledFunction } from './compile.ts';
import { EXTERNAL_OPS } from './externals.ts';
import {
  MODE_BUFFER,
  INITIAL_MODES,
  modeTableLength,
  randnfun3Chunks,
  randnfun3WGSL,
} from './randnfun3.ts';
import {
  buildKernel,
  UnsupportedOnGpu,
  WORKGROUP_SIZE,
  type KernelInputs,
} from './wgsl.ts';

const isNumeric = (t: Type): t is NumericType => t.kind === 'Numeric';
const isTensor = (t: Type): boolean => isNumeric(t) && isMultiElement(t);
const numel = (t: NumericType): number => (t.shape ?? []).reduce((a, b) => a * b, 1);

/** Scalar arithmetic a plan-time evaluator can fold. */
const PLAN_BINOPS: Record<string, (l: number, r: number) => number> = {
  plus: (l, r) => l + r,
  minus: (l, r) => l - r,
  times: (l, r) => l * r,
  mtimes: (l, r) => l * r,
  rdivide: (l, r) => l / r,
  mrdivide: (l, r) => l / r,
  power: (l, r) => Math.pow(l, r),
  mpower: (l, r) => Math.pow(l, r),
};

/** Scalar builtins of one argument the host can evaluate when a `lusolve`
 *  call's dtD or jhat is computed through them. */
const HOST_UNARY: Record<string, (x: number) => number> = {
  sqrt: Math.sqrt,
  exp: Math.exp,
  log: Math.log,
  abs: Math.abs,
};

/** A scalar the host evaluates from the current parameter values, in the
 *  params buffer's order — how the exact solver learns which factorization
 *  a step needs before it is encoded. */
export type HostScalar = (params: Float32Array) => number;

/** One `X = lusolve(B, dtD, ...)` call site, as planned. Its factorization
 *  (`site`) is attached once the model has built the operator. */
export interface ExactCall {
  B: Slot;
  X: Slot;
  dtD: HostScalar;
  jhat: HostScalar;
  label: string;
  site: ExactSite | null;
}

/** Cap on the iterations a `for` may unroll to. Each one is real GPU work —
 *  its own pipelines at compile time and its own dispatches per step — so a
 *  runaway bound should be a clear error rather than a hang. */
const MAX_UNROLL = 64;

interface Slot {
  buffer: GPUBuffer;
  count: number;
}

const makeBuffer = (device: GPUDevice, label: string, count: number): GPUBuffer =>
  device.createBuffer({
    label,
    size: 4 * count,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

/**
 * Buffers for host-bound variables, shared across plans.
 *
 * A model is two programs — `init` and `step` — compiled separately but
 * operating on the same state. `U` in the step must be the very buffer `init`
 * wrote, so the buffers for host bindings live here rather than inside either
 * plan.
 */
export class HostBuffers {
  #device: GPUDevice;
  #slots = new Map<string, Slot>();

  constructor(device: GPUDevice) {
    this.#device = device;
  }

  ensure(name: string, count: number): Slot {
    const existing = this.#slots.get(name);
    if (existing) {
      if (existing.count !== count) {
        throw new UnsupportedOnGpu(
          `'${name}' is ${existing.count} elements in one program and ` +
            `${count} in another`,
        );
      }
      return existing;
    }
    const slot = { buffer: makeBuffer(this.#device, `mgpu-${name}`, count), count };
    this.#slots.set(name, slot);
    return slot;
  }

  get(name: string): Slot | undefined {
    return this.#slots.get(name);
  }

  /**
   * Replace a slot's buffer with a larger one. Only for buffers whose size is
   * not fixed by the grid — the randnfun3 mode table, which grows with the
   * wavelength asked for. The caller must rebuild any bind group holding the
   * old buffer; it is destroyed here.
   */
  resize(name: string, count: number): Slot {
    const existing = this.#slots.get(name);
    if (!existing) throw new Error(`resize: no buffer named '${name}'`);
    if (count <= existing.count) return existing;
    existing.buffer.destroy();
    const slot = { buffer: makeBuffer(this.#device, `mgpu-${name}`, count), count };
    this.#slots.set(name, slot);
    return slot;
  }

  /** Upload into the front of a slot, leaving any tail as it was. For a
   *  variable-length payload in a buffer sized to its high-water mark. */
  uploadInto(name: string, data: Float32Array): void {
    const slot = this.#slots.get(name);
    if (!slot) throw new Error(`uploadInto: no buffer named '${name}'`);
    if (data.length > slot.count) {
      throw new Error(
        `uploadInto '${name}': ${data.length} elements into a ${slot.count}-element buffer`,
      );
    }
    this.#device.queue.writeBuffer(slot.buffer, 0, data as Float32Array<ArrayBuffer>);
  }

  /** Upload initial data for a host binding. */
  upload(name: string, data: Float32Array): void {
    const slot = this.#slots.get(name);
    if (!slot) throw new Error(`upload: no buffer named '${name}'`);
    if (data.length !== slot.count) {
      throw new Error(
        `upload '${name}': expected ${slot.count} elements, got ${data.length}`,
      );
    }
    this.#device.queue.writeBuffer(slot.buffer, 0, data as Float32Array<ArrayBuffer>);
  }

  destroy(): void {
    for (const s of this.#slots.values()) s.buffer.destroy();
    this.#slots.clear();
  }
}

type Op =
  | {
      kind: 'kernel';
      pipeline: GPUComputePipeline;
      bindGroup: GPUBindGroup;
      count: number;
      label: string;
      /** Set when the kernel had to write to scratch because its output
       *  aliases one of its inputs; copied back after the dispatch. */
      copyBack?: { from: GPUBuffer; to: GPUBuffer; bytes: number };
      /** End the submission here when run through `submitYielding`, so the
       *  GPU is handed back between chunks of a long seed. */
      yieldAfter?: boolean;
    }
  | { kind: 'synth' | 'analys'; binding: ShtBinding; label: string }
  | { kind: 'synth-batch' | 'analys-batch'; binding: ShtBatchBinding; labels: string[] }
  | { kind: 'dtheta' | 'dphi'; binding: DerivBinding; label: string }
  | { kind: 'dthetac' | 'dphic'; bindGroup: GPUBindGroup; label: string }
  | { kind: 'dphig'; binding: ShtDphigBinding; label: string }
  | { kind: 'dot'; binding: DotBinding; label: string }
  | { kind: 'lusolve'; call: ExactCall; label: string }
  | {
      kind: 'copy';
      from: GPUBuffer;
      to: GPUBuffer;
      bytes: number;
      label: string;
      /** Byte offsets, for the indexed-access ops. Absent means 0. */
      fromOffset?: number;
      toOffset?: number;
    };

/**
 * A transform op as planned, before bindings exist: `in`/`out` are the
 * caller-side buffers (spectral in / grid out for synth, the reverse for
 * analys). Kept unbound until every statement is planned so that adjacent
 * independent transforms of the same kind can be grouped into one batched
 * dispatch (ShtPlan.createSynthBatchBinding) — the Legendre recurrence is
 * the expensive shared part, and a batch walks it once for all lanes.
 */
interface PendingSht {
  pending: true;
  kind: 'synth' | 'analys';
  in: GPUBuffer;
  out: GPUBuffer;
  label: string;
}

type Planned = Op | PendingSht;

const isPending = (op: Planned): op is PendingSht => 'pending' in op;

/**
 * Group maximal runs of adjacent same-kind transforms into batches of the
 * widest compiled lane count, and create all bindings. Only literal
 * adjacency in the op sequence is batched — no reordering — so the models
 * are written to keep batchable transforms consecutive (see the solve loops
 * in models/*.m). Batching changes dispatch shape only: per-lane arithmetic
 * is identical to the scalar kernels', so results do not depend on batchK.
 */
function materializeTransforms(planned: Planned[], sht: ShtPlan): Op[] {
  /** Lanes must not collide: distinct outputs, and no lane reading another's
   *  output (repeated read-only inputs would be harmless, but WebGPU also
   *  forbids aliasing a writable binding, so outputs are the hard rule). */
  const disjoint = (members: PendingSht[]): boolean => {
    const outs = new Set<GPUBuffer>();
    for (const m of members) {
      if (outs.has(m.out)) return false;
      outs.add(m.out);
    }
    return members.every((m) => !outs.has(m.in));
  };
  const bind = (m: PendingSht): Op =>
    m.kind === 'synth'
      ? { kind: 'synth', binding: sht.createSynthBinding(m.in, m.out), label: m.label }
      : { kind: 'analys', binding: sht.createAnalysBinding(m.in, m.out), label: m.label };
  const bindBatch = (members: PendingSht[]): Op =>
    members[0].kind === 'synth'
      ? {
          kind: 'synth-batch',
          binding: sht.createSynthBatchBinding(
            members.map((m) => ({ qlmIn: m.in, spatOut: m.out })),
          ),
          labels: members.map((m) => m.label),
        }
      : {
          kind: 'analys-batch',
          binding: sht.createAnalysBatchBinding(
            members.map((m) => ({ spatIn: m.in, qlmOut: m.out })),
          ),
          labels: members.map((m) => m.label),
        };

  const out: Op[] = [];
  let i = 0;
  while (i < planned.length) {
    const op = planned[i];
    if (!isPending(op)) {
      out.push(op);
      i++;
      continue;
    }
    let j = i;
    while (j < planned.length) {
      const p = planned[j];
      if (!isPending(p) || p.kind !== op.kind) break;
      j++;
    }
    const run = planned.slice(i, j) as PendingSht[];
    let s = 0;
    while (s < run.length) {
      let take = 1;
      for (const K of [4, 2]) {
        if (K > sht.batchK || s + K > run.length) continue;
        if (disjoint(run.slice(s, s + K))) {
          take = K;
          break;
        }
      }
      out.push(take === 1 ? bind(run[s]) : bindBatch(run.slice(s, s + take)));
      s += take;
    }
    i = j;
  }
  return out;
}

export interface PlanSpec {
  /** The specialized function this plan executes. */
  fn: CompiledFunction;
  /** Output index -> host binding name to copy the result into after the run,
   *  so the next call reads it (the new spectral state feeds the old). */
  feedback: (string | null)[];
}

/**
 * Bind group layout for a kernel: the output at 0, `inputs` read-only storage
 * buffers after it, then the params buffer.
 *
 * Declared explicitly rather than with `layout: 'auto'`, because an auto layout
 * only contains the bindings the shader actually references — so a kernel that
 * happens to use no parameters (`uuv = u .* u .* v`) would drop the params
 * binding and no longer match the bind group. An explicit layout may carry
 * bindings the shader ignores.
 */
function kernelLayout(device: GPUDevice, inputs: number): GPUBindGroupLayout {
  const readOnly = (binding: number): GPUBindGroupLayoutEntry => ({
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: 'read-only-storage' },
  });
  return device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      ...Array.from({ length: inputs }, (_, i) => readOnly(i + 1)),
      readOnly(inputs + 1),
    ],
  });
}

async function makePipeline(
  device: GPUDevice,
  code: string,
  label: string,
  bindGroupLayout: GPUBindGroupLayout,
): Promise<GPUComputePipeline> {
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    throw new UnsupportedOnGpu(
      `generated WGSL failed to compile for '${label}':\n` +
        errors.map((e) => `  ${e.lineNum}:${e.linePos} ${e.message}`).join('\n') +
        `\n--- shader ---\n${code}`,
    );
  }
  const pipeline = await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module, entryPoint: 'main' },
    label,
  });
  const err = await device.popErrorScope();
  if (err) throw new UnsupportedOnGpu(`pipeline '${label}': ${err.message}`);
  return pipeline;
}

/** A compiled .m step, ready to run on the GPU. */
export class ModelPlan {
  /** Scalar parameter names, in the order the params buffer expects them. */
  readonly paramNames: string[];
  /** The wavelength this plan's `randnfun3` call asked for, or null if it
   *  makes none. The host draws the coefficient table from it. */
  readonly randnfun3Lambda: Randnfun3Lambda | null;

  #device: GPUDevice;
  #sht: ShtPlan;
  #deriv?: DerivPlan;
  #ops: Op[];
  #owned: GPUBuffer[];
  #paramBuf: GPUBuffer;
  #paramData: Float32Array;
  #rebindRandnfun3: ((table: GPUBuffer) => void) | null;
  /** Public name -> buffer, for uploading initial state and reading results. */
  #byName: Map<string, Slot>;
  /** The exact solver's call sites, in plan order. */
  #exactCalls: ExactCall[];

  private constructor(init: {
    device: GPUDevice;
    sht: ShtPlan;
    deriv?: DerivPlan;
    ops: Op[];
    byName: Map<string, Slot>;
    owned: GPUBuffer[];
    paramBuf: GPUBuffer;
    paramData: Float32Array;
    paramNames: string[];
    randnfun3Lambda: Randnfun3Lambda | null;
    rebindRandnfun3: ((table: GPUBuffer) => void) | null;
    exactCalls: ExactCall[];
  }) {
    this.#device = init.device;
    this.#sht = init.sht;
    this.#deriv = init.deriv;
    this.#ops = init.ops;
    this.#byName = init.byName;
    this.#owned = init.owned;
    this.#paramBuf = init.paramBuf;
    this.#paramData = init.paramData;
    this.paramNames = init.paramNames;
    this.randnfun3Lambda = init.randnfun3Lambda;
    this.#rebindRandnfun3 = init.rebindRandnfun3;
    this.#exactCalls = init.exactCalls;
  }

  /** The `lusolve` call sites this plan makes — non-empty means the model
   *  needs an ExactOperator attached before it can run. */
  get exactCalls(): readonly ExactCall[] {
    return this.#exactCalls;
  }

  /** Give every `lusolve` site its factorization against `op`, the
   *  operator built for this model's grid and surface. */
  attachExact(op: ExactOperator): void {
    for (const call of this.#exactCalls) {
      call.site?.destroy();
      call.site = op.createSite(call.B.buffer, call.X.buffer, call.label);
    }
  }

  /**
   * Before a run is recorded: the exact solver's factorizations must hold
   * the (dtD, jhat) this run will use, and the current operator. The host
   * evaluates both scalars from the parameter values it last uploaded — the
   * same values the kernels read — and (re)factors when they moved, in the
   * same submission, ahead of the step. Cheap when nothing changed.
   */
  #prepareExact(encoder: GPUCommandEncoder): void {
    for (const call of this.#exactCalls) {
      if (!call.site) {
        throw new Error(
          `'${call.label}': the exact solver's operator was never built for this plan`,
        );
      }
      // Rounded as the kernels would compute them, in fp32.
      const dtD = Math.fround(call.dtD(this.#paramData));
      const jhat = Math.fround(call.jhat(this.#paramData));
      call.site.ensureFactored(encoder, dtD, jhat);
    }
  }

  /**
   * Point the randnfun3 dispatch at a mode table big enough for `data`,
   * growing the buffer if this wavelength needs more modes than the last one,
   * and upload it.
   */
  uploadRandnfun3Table(host: HostBuffers, data: Float32Array): void {
    const slot = host.get(MODE_BUFFER);
    if (!slot || !this.#rebindRandnfun3) return;
    if (data.length > slot.count) {
      const max = this.#device.limits.maxStorageBufferBindingSize;
      if (4 * data.length > max) {
        throw new Error(
          `randnfun3: this wavelength needs a ${(4 * data.length / 1e6).toFixed(0)} MB ` +
            `mode table, past this device's ${(max / 1e6).toFixed(0)} MB limit ` +
            `on a single buffer. Use a larger lambda.`,
        );
      }
      this.#rebindRandnfun3(host.resize(MODE_BUFFER, data.length).buffer);
    }
    host.uploadInto(MODE_BUFFER, data);
  }

  static async create(
    device: GPUDevice,
    sht: ShtPlan,
    spec: PlanSpec,
    host: HostBuffers,
    /** Computes dtheta/dphi — only needed if the .m calls them. */
    deriv?: DerivPlan,
  ): Promise<ModelPlan> {
    const { fn } = spec;

    const slots = new Map<string, Slot>();
    const byName = new Map<string, Slot>();
    const owned: GPUBuffer[] = [];
    /** Scalars the .m computes from its parameters, by cName. */
    const derivedScalars = new Map<string, { name: string; expr: IRExpr }>();
    const exactCalls: ExactCall[] = [];

    const alloc = (label: string, count: number): Slot => {
      const buffer = makeBuffer(device, label, count);
      owned.push(buffer);
      return { buffer, count };
    };

    // Arguments, bound by what the function's signature declares. Array
    // arguments come from the shared pool, so a value one function returns is
    // the same buffer the next one reads. Scalar parameters share one small
    // storage buffer, in signature order.
    const paramNames: string[] = [];
    const paramSlots = new Map<string, number>();
    for (const p of fn.params) {
      if (p.binding.kind === 'tensor') {
        const count = p.binding.shape.reduce((x, y) => x * y, 1);
        const slot = host.ensure(p.name, count);
        slots.set(p.cName, slot);
        byName.set(p.name, slot);
      } else if (p.binding.kind === 'param') {
        paramSlots.set(p.cName, paramNames.length);
        paramNames.push(p.name);
      }
      // `const` arguments are exact in the IR and fold into the kernels.
    }
    const paramData = new Float32Array(Math.max(1, paramNames.length));
    const paramBuf = device.createBuffer({
      label: 'mgpu-params',
      size: 4 * paramData.length,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    /** Set when the .m calls `randnfun3`: which wavelength it asked for, so
     *  the host draws the coefficient table the kernel reads from exactly
     *  that value (src/mgpu/randnfun3.ts). */
    let randnfun3Lambda: Randnfun3Lambda | null = null;
    /** Rebuilds the randnfun3 dispatch's bind group after the mode table is
     *  reallocated for a finer wavelength. */
    let rebindRandnfun3: ((table: GPUBuffer) => void) | null = null;

    /** Built on first use — only a model that calls `dot` pays for it. */
    let reduce: ReducePlan | null = null;

    /** Does this expression read any GPU-resident value? Decides whether a
     *  scalar assignment can stay a compile-time derived scalar or needs a
     *  1-element kernel. Plan-order matters and is correct: a name is
     *  buffer-backed from the statement that first computes it into one. */
    const readsBufferValue = (e: IRExpr): boolean => {
      let found = false;
      collectVars(e, (v) => {
        if (slots.has(v.cName)) found = true;
      });
      return found;
    };

    /**
     * The value a scalar expression has *at this point in the plan*, if it
     * is decidable. A literal carries its own; a variable carries one via
     * numbl's `exact` lattice or — the case the lattice cannot see — via its
     * derived-scalar binding, which is how an unrolled loop's variable (and
     * anything computed from it, like an index or an inner loop bound)
     * resolves to that iteration's literal. A buffer-backed name is a
     * runtime value and never resolves.
     */
    const planTimeValue = (e: IRExpr): number | undefined => {
      if (e.kind === 'NumLit') return e.value;
      if (isNumeric(e.ty) && typeof e.ty.exact === 'number') return e.ty.exact;
      switch (e.kind) {
        case 'Var': {
          if (slots.has(e.cName)) return undefined;
          const d = derivedScalars.get(e.cName);
          return d ? planTimeValue(d.expr) : undefined;
        }
        case 'Binary': {
          const op = PLAN_BINOPS[e.builtin];
          if (!op) return undefined;
          const l = planTimeValue(e.left);
          const r = planTimeValue(e.right);
          return l === undefined || r === undefined ? undefined : op(l, r);
        }
        case 'Unary': {
          const v = planTimeValue(e.operand);
          if (v === undefined) return undefined;
          if (e.builtin === 'uminus') return -v;
          if (e.builtin === 'uplus') return v;
          return undefined;
        }
        default:
          return undefined;
      }
    };

    /**
     * A scalar as a function of the parameter values, if the host can
     * evaluate it: literals, exact constants, parameters, scalars derived
     * from them through the arithmetic the plan folds and a few one-argument
     * builtins. A buffer-backed value (a `dot` result, or anything downstream
     * of one) is GPU-resident and cannot be evaluated here.
     */
    const hostScalar = (e: IRExpr): HostScalar | undefined => {
      if (e.kind === 'NumLit') {
        const v = e.value;
        return () => v;
      }
      if (isNumeric(e.ty) && typeof e.ty.exact === 'number') {
        const v = e.ty.exact;
        return () => v;
      }
      switch (e.kind) {
        case 'Var': {
          if (slots.has(e.cName)) return undefined;
          const p = paramSlots.get(e.cName);
          if (p !== undefined) return (params) => params[p];
          const d = derivedScalars.get(e.cName);
          return d ? hostScalar(d.expr) : undefined;
        }
        case 'Binary': {
          const op = PLAN_BINOPS[e.builtin];
          if (!op) return undefined;
          const l = hostScalar(e.left);
          const r = hostScalar(e.right);
          return l && r ? (params) => op(l(params), r(params)) : undefined;
        }
        case 'Unary': {
          const v = hostScalar(e.operand);
          if (!v) return undefined;
          if (e.builtin === 'uminus') return (params) => -v(params);
          if (e.builtin === 'uplus') return v;
          return undefined;
        }
        case 'Call': {
          const f = HOST_UNARY[e.name];
          if (!f || e.args.length !== 1) return undefined;
          const a = hostScalar(e.args[0]);
          return a ? (params) => f(a(params)) : undefined;
        }
        default:
          return undefined;
      }
    };

    /** A plan-time index: integral and 1-based. */
    const planTimeIndex = (e: IRExpr, what: string, span: unknown): number => {
      const v = planTimeValue(e);
      if (v === undefined) {
        throw new UnsupportedOnGpu(
          `${what} must be known when the model compiles — a literal, a fixed ` +
            `argument, or a value of the unrolled loop's variable`,
          span,
        );
      }
      if (!Number.isInteger(v) || v < 1) {
        throw new UnsupportedOnGpu(`${what} must be a positive integer (got ${v})`, span);
      }
      return v;
    };

    const planned: Planned[] = [];
    for (const stmt of fn.body) {
      await planStatement(stmt);
    }

    // Feed declared outputs back into the argument buffers they replace.
    fn.outputs.forEach((out, i) => {
      const to = spec.feedback[i];
      if (!to) return;
      const src = slots.get(out.cName);
      const dst = host.get(to);
      if (!src) {
        throw new UnsupportedOnGpu(
          `'${fn.name}' declares the output '${out.name}' but never assigns it`,
        );
      }
      if (!dst) throw new UnsupportedOnGpu(`'${to}' is not a host binding`);
      if (src.count !== dst.count) {
        throw new UnsupportedOnGpu(
          `'${out.name}' (${src.count} elements) cannot feed ` +
            `'${to}' (${dst.count})`,
        );
      }
      planned.push({
        kind: 'copy',
        from: src.buffer,
        to: dst.buffer,
        bytes: 4 * src.count,
        label: `${out.name} -> ${to}`,
      });
    });

    // Group adjacent independent transforms into batched dispatches and
    // create every binding.
    const ops = materializeTransforms(planned, sht);

    return new ModelPlan({
      device, sht, deriv, ops, byName, owned, paramBuf, paramData, paramNames,
      randnfun3Lambda, rebindRandnfun3, exactCalls,
    });

    async function planStatement(stmt: IRStmt): Promise<void> {
      if (stmt.kind === 'ReturnFromFunction') return; // nothing follows it
      if (stmt.kind === 'For') return planFor(stmt);
      if (stmt.kind === 'MultiAssignCall') return planMultiTransform(stmt);
      if (stmt.kind !== 'Assign') {
        throw new UnsupportedOnGpu(
          `a model function body may only contain assignments ` +
            `(found '${stmt.kind}')`,
          stmt.span,
        );
      }
      if (!isNumeric(stmt.ty)) {
        throw new UnsupportedOnGpu(
          `'${stmt.name}' is not a numeric value`,
          stmt.span,
        );
      }
      const ext = externalCall(stmt);
      if (!isTensor(stmt.ty) && !ext && !readsBufferValue(stmt.expr)) {
        // A scalar the model derives from its parameters (`us = a + b`). It
        // gets no buffer and no dispatch: the kernels that read it bind it as
        // a `let` in their prologue. A scalar computed from GPU-resident
        // values (a `dot` result, or anything downstream of one) instead
        // falls through to a 1-element kernel, because its inputs live in
        // buffers the CPU never sees.
        derivedScalars.set(stmt.cName, { name: stmt.name, expr: stmt.expr });
        return;
      }
      const count = numel(stmt.ty);

      // Reuse the destination buffer across steps: the same cName always maps
      // to the same buffer, so a step allocates nothing.
      let dest = slots.get(stmt.cName);
      if (!dest) {
        dest = alloc(`mgpu-${stmt.name}`, count);
        slots.set(stmt.cName, dest);
      } else if (dest.count !== count) {
        throw new UnsupportedOnGpu(
          `'${stmt.name}' changes size between assignments`,
          stmt.span,
        );
      }
      byName.set(stmt.name, dest);

      if (ext) {
        if (ext.name === 'randnfun3') {
          await planRandnfun3(stmt, ext.args, dest);
          return;
        }
        // Lazy per-argument resolution: buffer arguments must have slots,
        // while index arguments are plan-time scalars with no buffer at all.
        const argSlot = (i: number): Slot => {
          const a = ext.args[i];
          if (a.kind !== 'Var') {
            throw new UnsupportedOnGpu(
              `'${ext.name}' needs a plain variable here — assign the ` +
                `expression to a variable first`,
              stmt.span,
            );
          }
          const s = slots.get(a.cName);
          if (!s) {
            throw new UnsupportedOnGpu(
              `'${ext.name}' reads '${a.name}', which has no buffer`,
              stmt.span,
            );
          }
          return s;
        };
        const label = `${stmt.name} = ${ext.name}(${ext.args.map(extArgName).join(', ')})`;
        if (ext.name === 'dot') {
          const a = argSlot(0);
          const b = argSlot(1);
          if (a.count !== b.count) {
            throw new UnsupportedOnGpu(
              `'dot' needs equal-length arguments (${a.count} vs ${b.count})`,
              stmt.span,
            );
          }
          if (a.buffer === dest.buffer || b.buffer === dest.buffer) {
            throw new UnsupportedOnGpu(
              `'dot' cannot write over one of its own arguments`,
              stmt.span,
            );
          }
          reduce ??= new ReducePlan(device);
          planned.push({
            kind: 'dot',
            binding: await reduce.createDotBinding(a.buffer, b.buffer, dest.buffer, a.count),
            label,
          });
          return;
        }
        if (ext.name === 'lusolve') {
          planLusolve(stmt, ext.args, dest, argSlot, label);
          return;
        }
        if (ext.name === 'getslab' || ext.name === 'setslab') {
          const slabElems = 2 * sht.nlm;
          const bank = argSlot(0);
          const nslabs = Math.floor(bank.count / slabElems);
          const kArg = ext.args[ext.name === 'getslab' ? 1 : 2];
          const k = planTimeIndex(kArg, `'${ext.name}'s index '${extArgName(kArg)}'`, stmt.span);
          if (bank.count % slabElems !== 0 || k > nslabs) {
            throw new UnsupportedOnGpu(
              `'${ext.name}': slab ${k} is out of range for a bank of ` +
                `${nslabs} spectral fields`,
              stmt.span,
            );
          }
          const slabBytes = 4 * slabElems;
          if (ext.name === 'getslab') {
            if (dest.count !== slabElems || bank.buffer === dest.buffer) {
              throw new UnsupportedOnGpu(`'getslab' cannot read into its own bank`, stmt.span);
            }
            planned.push({
              kind: 'copy', from: bank.buffer, fromOffset: (k - 1) * slabBytes,
              to: dest.buffer, bytes: slabBytes, label,
            });
          } else {
            const field = argSlot(1);
            if (field.count !== slabElems || field.buffer === dest.buffer) {
              throw new UnsupportedOnGpu(
                `'setslab' needs a distinct 2 x nlm field to write`,
                stmt.span,
              );
            }
            // Functional update: writing back over the base is the in-place
            // fast path; a fresh destination first takes a copy of the bank.
            if (dest.buffer !== bank.buffer) {
              planned.push({
                kind: 'copy', from: bank.buffer, to: dest.buffer,
                bytes: 4 * bank.count, label: `${label} (bank copy)`,
              });
            }
            planned.push({
              kind: 'copy', from: field.buffer,
              to: dest.buffer, toOffset: (k - 1) * slabBytes,
              bytes: slabBytes, label,
            });
          }
          return;
        }
        if (ext.name === 'getat' || ext.name === 'setat') {
          const base = argSlot(0);
          const baseTy = ext.args[0].ty;
          if (ext.args[0].kind !== 'Var') {
            throw new UnsupportedOnGpu(`'${ext.name}' needs a variable base`, stmt.span);
          }
          const shape = isNumeric(baseTy) ? baseTy.shape : undefined;
          if (!shape) {
            throw new UnsupportedOnGpu(`'${ext.name}' needs a base of known shape`, stmt.span);
          }
          const idxArgs = ext.args.slice(ext.name === 'getat' ? 1 : 2);
          const idx = idxArgs.map(
            (a) => planTimeIndex(a, `'${ext.name}'s index '${extArgName(a)}'`, stmt.span) - 1,
          );
          // Column-major, like everything else in the 2 x nlm layout: a
          // 2-index access is (i-1) + (j-1)*rows, a 1-index access is linear.
          let offset: number;
          if (idx.length === 2) {
            const [i, j] = idx;
            if (i >= shape[0] || j >= (shape[1] ?? 1)) {
              throw new UnsupportedOnGpu(
                `'${ext.name}': (${i + 1}, ${j + 1}) is outside ` +
                  `${shape.join('x')} '${extArgName(ext.args[0])}'`,
                stmt.span,
              );
            }
            offset = i + j * shape[0];
          } else {
            offset = idx[0];
            if (offset >= base.count) {
              throw new UnsupportedOnGpu(
                `'${ext.name}': index ${offset + 1} is outside ` +
                  `${base.count}-element '${extArgName(ext.args[0])}'`,
                stmt.span,
              );
            }
          }
          if (ext.name === 'getat') {
            if (dest.count !== 1 || base.buffer === dest.buffer) {
              throw new UnsupportedOnGpu(`'getat' cannot read into its own base`, stmt.span);
            }
            planned.push({
              kind: 'copy', from: base.buffer, fromOffset: 4 * offset,
              to: dest.buffer, bytes: 4, label,
            });
          } else {
            const value = argSlot(1);
            if (value.count !== 1 || value.buffer === dest.buffer) {
              throw new UnsupportedOnGpu(
                `'setat' needs a distinct 1-element value to write — compute ` +
                  `it into a variable first`,
                stmt.span,
              );
            }
            if (dest.buffer !== base.buffer) {
              planned.push({
                kind: 'copy', from: base.buffer, to: dest.buffer,
                bytes: 4 * base.count, label: `${label} (base copy)`,
              });
            }
            planned.push({
              kind: 'copy', from: value.buffer,
              to: dest.buffer, toOffset: 4 * offset, bytes: 4, label,
            });
          }
          return;
        }
        const src = argSlot(0);
        if (ext.name === 'dphig') {
          // Grid -> grid, staged through the plan's fm scratch; safe even
          // in place, so no aliasing guard is needed.
          planned.push({
            kind: 'dphig',
            binding: sht.createDphigBinding(src.buffer, dest.buffer),
            label,
          });
          return;
        }
        if (ext.name === 'synth' || ext.name === 'analys') {
          // Left unbound until materializeTransforms has grouped adjacent
          // independent transforms into batched dispatches.
          planned.push({
            pending: true,
            kind: ext.name,
            in: src.buffer,
            out: dest.buffer,
            label,
          });
        } else if (
          ext.name === 'dtheta' || ext.name === 'dphi' ||
          ext.name === 'dthetac' || ext.name === 'dphic'
        ) {
          if (!deriv) {
            throw new UnsupportedOnGpu(
              `'${ext.name}' needs the surface's derivative transforms, ` +
                `which this plan was not given`,
              stmt.span,
            );
          }
          if (ext.name === 'dthetac' || ext.name === 'dphic') {
            // Coefficient-space shuffles read at l+-1 (dthetac) or in place
            // (dphic) and cannot alias their output: WebGPU forbids one buffer
            // being readable and writable storage in the same dispatch, and
            // there is no scratch-copy fallback here — refuse rather than
            // silently reroute.
            if (src.buffer === dest.buffer) {
              throw new UnsupportedOnGpu(
                `'${label}' reads and ` +
                  `writes the same buffer; assign to a new name instead`,
                stmt.span,
              );
            }
            planned.push({
              kind: ext.name,
              bindGroup:
                ext.name === 'dthetac'
                  ? deriv.createDthetacBinding(src.buffer, dest.buffer)
                  : deriv.createDphicBinding(src.buffer, dest.buffer),
              label,
            });
            return;
          }
          planned.push(
            ext.name === 'dtheta'
              ? { kind: 'dtheta', binding: deriv.createDthetaBinding(src.buffer, dest.buffer), label }
              : { kind: 'dphi', binding: deriv.createDphiBinding(src.buffer, dest.buffer), label },
          );
        } else {
          throw new UnsupportedOnGpu(`unknown external op '${ext.name}'`, stmt.span);
        }
        return;
      }

      // Element-wise kernel. Collect the distinct buffer-backed operands —
      // multi-element tensors, plus any single-element value living in a
      // buffer (a dot result or a scalar computed from one) — and give them
      // dense binding slots. The kernel reads a single-element operand as
      // `in<slot>[0]`, which is what broadcasts it across the output.
      const tensors = new Map<string, number>();
      collectVars(stmt.expr, (v) => {
        if (!isTensor(v.ty) && !slots.has(v.cName)) return;
        if (!tensors.has(v.cName)) tensors.set(v.cName, tensors.size);
      });

      const label = `${stmt.name} = <${count} elements, element-wise>`;
      const kernel = buildKernel(
        stmt,
        {
          tensors,
          params: paramSlots,
          scalars: derivedScalars,
        } satisfies KernelInputs,
        count,
        label,
      );

      const bindGroupLayout = kernelLayout(device, tensors.size);
      const pipeline = await makePipeline(device, kernel.code, label, bindGroupLayout);

      // WebGPU forbids aliasing a writable storage binding with another
      // binding in the same group, so an in-place update (`u = u + 1`) writes
      // to scratch and copies back. Element-wise kernels only ever touch
      // their own index, so the copy is the only cost.
      const aliased = tensors.has(stmt.cName);
      const target = aliased ? alloc(`mgpu-${stmt.name}-scratch`, count) : dest;

      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: target.buffer } },
      ];
      for (const [cName, i] of tensors) {
        const s = slots.get(cName);
        if (!s) {
          throw new UnsupportedOnGpu(
            `'${stmt.name}' reads a value with no buffer`,
            stmt.span,
          );
        }
        entries.push({ binding: i + 1, resource: { buffer: s.buffer } });
      }
      entries.push({ binding: tensors.size + 1, resource: { buffer: paramBuf } });

      planned.push({
        kind: 'kernel',
        pipeline,
        bindGroup: device.createBindGroup({
          layout: bindGroupLayout,
          entries,
        }),
        count,
        label,
        copyBack: aliased
          ? { from: target.buffer, to: dest.buffer, bytes: 4 * count }
          : undefined,
      });
    }

    /**
     * `X = lusolve(B, dtD, lam, filt, jhat, p2, r, dp1, dq2, jinv)`: the exact
     * solver's dense direct solve (src/mgpu/exact.ts). What is planned here
     * is the call site — its buffers and how the host reads dtD and jhat —
     * and the checks that the matrix the host will assemble is the matrix
     * this call means: the operator arguments must be the app's own arrays,
     * because the column program that builds K reads those by name. The
     * factorization itself is attached by the model once the operator is
     * built (`attachExact`), and refreshed per run by `#prepareExact`.
     */
    function planLusolve(
      stmt: Assign,
      args: IRExpr[],
      dest: Slot,
      argSlot: (i: number) => Slot,
      label: string,
    ): void {
      const n = 2 * sht.nlm;
      const B = argSlot(0);
      if (B.count !== n || dest.count !== n) {
        throw new UnsupportedOnGpu(`'lusolve' solves for a 2 x nlm spectral field`, stmt.span);
      }
      if (B.buffer === dest.buffer) {
        throw new UnsupportedOnGpu(
          `'lusolve' cannot write over its right-hand side; assign to a new name`,
          stmt.span,
        );
      }
      const scalar = (i: number, what: string): HostScalar => {
        const f = hostScalar(args[i]);
        if (!f) {
          throw new UnsupportedOnGpu(
            `'lusolve' factors its matrix for the value of ${what} before the step ` +
              `runs, so ${what} must be computable from the model's parameters — ` +
              `not from a value computed on the GPU`,
            stmt.span,
          );
        }
        return f;
      };
      const dtD = scalar(1, 'dtD');
      const jhat = scalar(4, 'jhat');
      const operatorArgs: [number, string][] = [
        [2, 'lam'], [3, 'filt'], [5, 'p2'], [6, 'r'], [7, 'dp1'], [8, 'dq2'], [9, 'jinv'],
      ];
      for (const [i, name] of operatorArgs) {
        if (argSlot(i).buffer !== host.get(name)?.buffer) {
          throw new UnsupportedOnGpu(
            `'lusolve' assembles its matrix from the app's own '${name}' array, ` +
              `so that argument must be '${name}' itself, passed through unchanged ` +
              `(got '${extArgName(args[i])}')`,
            stmt.span,
          );
        }
      }
      const bytes = exactMatrixBytes(n);
      const limit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
      if (bytes > limit) {
        throw new UnsupportedOnGpu(
          `the exact solver keeps the operator as a dense ${n} x ${n} matrix — ` +
            `${(bytes / 1e6).toFixed(0)} MB at this band, over this device's ` +
            `${(limit / 1e6).toFixed(0)} MB limit on one buffer. Use a smaller lmax ` +
            `or an iterative solver.`,
          stmt.span,
        );
      }
      const call: ExactCall = { B, X: dest, dtD, jhat, label, site: null };
      exactCalls.push(call);
      planned.push({ kind: 'lusolve', call, label });
    }

    /**
     * `[a, b] = synth(x, y)` / `[a, b] = analys(x, y)`: an explicitly grouped
     * transform — output k is the transform of argument k. The group is
     * planned as consecutive pending transforms, which materializeTransforms
     * then chunks into whatever batched dispatch widths the device supports
     * (one x4 batch, two x2, or scalars with SHT_BATCH=0) — the syntax
     * promises grouping intent, never a lane width, so the same source
     * compiles everywhere.
     */
    function planMultiTransform(stmt: MultiAssignCall): void {
      if (stmt.name !== 'synth' && stmt.name !== 'analys') {
        throw new UnsupportedOnGpu(
          `'${stmt.name}' does not return multiple values here — only the ` +
            `transforms ('synth', 'analys') support [a, b] = op(x, y) grouping`,
          stmt.span,
        );
      }
      const kind = stmt.name;
      for (let i = 0; i < stmt.outputs.length; i++) {
        const slot = stmt.outputs[i];
        const arg = stmt.args[i];
        if (!slot.binding) {
          throw new UnsupportedOnGpu(
            `every output of '${kind}' must be bound to a name — output ` +
              `${i + 1} is dropped, but each input costs a transform`,
            stmt.span,
          );
        }
        if (!arg || arg.kind !== 'Var') {
          throw new UnsupportedOnGpu(
            `'${kind}' must be applied to variables (argument ${i + 1})`,
            stmt.span,
          );
        }
        const argSlot = slots.get(arg.cName);
        if (!argSlot) {
          throw new UnsupportedOnGpu(
            `'${kind}' reads '${arg.name}', which has no buffer`,
            stmt.span,
          );
        }
        if (!isNumeric(slot.ty) || !isTensor(slot.ty)) {
          throw new UnsupportedOnGpu(
            `'${slot.binding.name}' is not a numeric array`,
            stmt.span,
          );
        }
        const count = numel(slot.ty);
        let dest = slots.get(slot.binding.cName);
        if (!dest) {
          dest = alloc(`mgpu-${slot.binding.name}`, count);
          slots.set(slot.binding.cName, dest);
        } else if (dest.count !== count) {
          throw new UnsupportedOnGpu(
            `'${slot.binding.name}' changes size between assignments`,
            stmt.span,
          );
        }
        byName.set(slot.binding.name, dest);
        planned.push({
          pending: true,
          kind,
          in: argSlot.buffer,
          out: dest.buffer,
          label: `${slot.binding.name} = ${kind}(${arg.name})`,
        });
      }
    }

    /**
     * `f = randnfun3(lambda, gx, gy, gz)`: the seeded random field, summed
     * over its Fourier modes at every surface point.
     *
     * One dispatch, one thread per point. The coefficient table is not an
     * argument — it is a host buffer this plan binds and the host refills per
     * seed, the way `synth` reads Legendre matrices the .m never names. What
     * the .m *does* choose is the wavelength, which is recorded here so the
     * host draws the table for exactly that value.
     */
    async function planRandnfun3(
      stmt: Assign,
      args: IRExpr[],
      dest: Slot,
    ): Promise<void> {
      const lam = args[0];
      const lambda: Randnfun3Lambda | null =
        lam.kind === 'NumLit'
          ? { kind: 'const', value: lam.value }
          : lam.kind === 'Var' && paramSlots.has(lam.cName)
            ? { kind: 'param', name: lam.name }
            : null;
      if (!lambda) {
        throw new UnsupportedOnGpu(
          `randnfun3's wavelength is drawn on the host before the step runs, ` +
            `so it must be a number or a model parameter — not a value ` +
            `computed on the GPU`,
          stmt.span,
        );
      }
      if (randnfun3Lambda && !sameLambda(randnfun3Lambda, lambda)) {
        throw new UnsupportedOnGpu(
          `this function calls randnfun3 with two different wavelengths; ` +
            `one coefficient table is drawn per plan, so only one is supported`,
          stmt.span,
        );
      }
      randnfun3Lambda = lambda;

      const points = args.slice(1).map((a) => {
        const v = a as IRExpr & { kind: 'Var' };
        const slot = slots.get(v.cName);
        if (!slot) {
          throw new UnsupportedOnGpu(
            `randnfun3 reads '${v.name}', which has no buffer`,
            stmt.span,
          );
        }
        return { slot, name: v.name };
      });

      const modes = host.ensure(MODE_BUFFER, modeTableLength(INITIAL_MODES));
      const label =
        `${stmt.name} = randnfun3(${
          lambda.kind === 'const' ? lambda.value : lambda.name
        }, ${points.map((p) => p.name).join(', ')})`;

      const bindGroupLayout = device.createBindGroupLayout({
        label: 'mgpu-randnfun3',
        entries: [0, 1, 2, 3, 4].map((binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: binding === 0 ? ('storage' as const) : ('read-only-storage' as const) },
        })),
      });
      // The table is sized to whatever wavelength is actually asked for, so a
      // finer one reallocates it — and with it these bind groups, which are
      // the only things holding the old buffer.
      const bind = (table: GPUBuffer): GPUBindGroup =>
        device.createBindGroup({
          layout: bindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: dest.buffer } },
            ...points.map((p, i) => ({
              binding: i + 1,
              resource: { buffer: p.slot.buffer },
            })),
            { binding: 4, resource: { buffer: table } },
          ],
        });

      // One dispatch per slice of the mode table — see randnfun3Chunks. Each
      // reads the same table and accumulates into the same output, so they
      // share a bind group and differ only in their compiled slice index.
      const ops: (Op & { kind: 'kernel' })[] = [];
      for (let chunk = 0; chunk < randnfun3Chunks; chunk++) {
        const chunkLabel = `${label} [${chunk + 1}/${randnfun3Chunks}]`;
        const op = {
          kind: 'kernel' as const,
          pipeline: await makePipeline(
            device,
            randnfun3WGSL(dest.count, chunk),
            chunkLabel,
            bindGroupLayout,
          ),
          bindGroup: bind(modes.buffer),
          count: dest.count,
          label: chunkLabel,
          yieldAfter: true,
        };
        ops.push(op);
        planned.push(op);
      }
      rebindRandnfun3 = (table: GPUBuffer): void => {
        const group = bind(table);
        for (const op of ops) op.bindGroup = group;
      };
    }

    /**
     * Unroll a counted loop into the op sequence.
     *
     * A plan is a fixed list of GPU operations with no branching, which is what
     * makes a timestep pure command recording. A `for` with compile-time-known
     * bounds still fits that: it is the same body planned once per iteration.
     * Nothing else changes — numbl gives a variable one cName for every
     * assignment to it, so the buffer an iteration writes is the buffer the
     * next one reads, which is exactly a loop-carried value.
     *
     * The loop variable gets no buffer either: it is bound as a derived scalar
     * to this iteration's literal value, so a kernel that reads `k` folds the
     * number in. The binding is overwritten per iteration, before that
     * iteration's body is planned and its WGSL emitted.
     */
    async function planFor(stmt: For): Promise<void> {
      const from = planTimeValue(stmt.start);
      const to = planTimeValue(stmt.end);
      if (from === undefined || to === undefined) {
        throw new UnsupportedOnGpu(
          `a 'for' loop is unrolled into the op sequence, so its bounds must ` +
            `be known when the model is compiled — ` +
            `${from === undefined ? 'the start' : 'the end'} of this one is a ` +
            `runtime value. Use a whole number, a count the app supplies ` +
            `as a fixed argument (changing it recompiles), or an enclosing ` +
            `unrolled loop's variable.`,
          stmt.span,
        );
      }
      const trips = Math.floor((to - from) / stmt.step) + 1;
      if (!Number.isFinite(trips)) {
        throw new UnsupportedOnGpu(`'for ${stmt.varName}' has no finite length`, stmt.span);
      }
      if (trips > MAX_UNROLL) {
        throw new UnsupportedOnGpu(
          `'for ${stmt.varName}' would unroll to ${trips} iterations, over the ` +
            `limit of ${MAX_UNROLL}. Every iteration is separate GPU work, so a ` +
            `long loop compiles slowly and runs no faster than writing it out.`,
          stmt.span,
        );
      }
      for (let i = 0; i < trips; i++) {
        const value = from + i * stmt.step;
        derivedScalars.set(stmt.cVar, {
          name: stmt.varName,
          expr: {
            kind: 'NumLit',
            value,
            ty: scalarDouble(
              value > 0 ? 'positive' : value < 0 ? 'negative' : 'zero',
              value,
            ),
            span: stmt.span,
          },
        });
        for (const s of stmt.body) await planStatement(s);
      }
    }
  }

  /** Upload parameter values, in `paramNames` order. Cheap — call freely. */
  setParams(values: Record<string, number>): void {
    this.paramNames.forEach((name, i) => {
      const v = values[name];
      this.#paramData[i] = Number.isFinite(v) ? v : 0;
    });
    this.#device.queue.writeBuffer(
      this.#paramBuf,
      0,
      this.#paramData as Float32Array<ArrayBuffer>,
    );
  }

  /** Buffer holding the named value, or undefined if the .m never binds it. */
  buffer(name: string): GPUBuffer | undefined {
    return this.#byName.get(name)?.buffer;
  }

  elementCount(name: string): number | undefined {
    return this.#byName.get(name)?.count;
  }

  /**
   * Run one pass of this plan, submitting in pieces so the GPU is not held for
   * the whole of it.
   *
   * For `init` only, and only because the seed field's mode sum can be huge:
   * at a fine wavelength the dispatches add up to tens of seconds, and a
   * browser's GPU process is shared with compositing, so one submission that
   * long stops the whole browser painting — the user's tabs included. Ops
   * marked `yieldAfter` (the randnfun3 chunks) end their submission and give
   * the queue back before the next one is recorded, which turns a freeze into
   * a wait. Everything else is recorded exactly as `encodeSteps` would.
   */
  async submitYielding(label: string): Promise<void> {
    let encoder = this.#device.createCommandEncoder({ label });
    this.#prepareExact(encoder);
    let any = false;
    for (const group of this.#yieldGroups()) {
      if (any) {
        // Let the queue drain, then hand the event loop back, so compositing
        // and input get a turn between chunks.
        await this.#device.queue.onSubmittedWorkDone();
        await new Promise((r) => setTimeout(r, 0));
        encoder = this.#device.createCommandEncoder({ label });
      }
      this.#encodeOps(encoder, group);
      this.#device.queue.submit([encoder.finish()]);
      any = true;
    }
    if (!any) {
      this.#encodeOps(encoder, []);
      this.#device.queue.submit([encoder.finish()]);
    }
  }

  /** The op list split at every `yieldAfter` boundary. */
  *#yieldGroups(): Generator<Op[]> {
    let group: Op[] = [];
    for (const op of this.#ops) {
      group.push(op);
      if (op.kind === 'kernel' && op.yieldAfter) {
        yield group;
        group = [];
      }
    }
    if (group.length) yield group;
  }

  /**
   * Record `steps` timesteps. Synchronous: no awaits, no readback. All of the
   * ops share one compute pass, which WebGPU executes in submission order
   * with a barrier between dispatches.
   */
  encodeSteps(encoder: GPUCommandEncoder, steps: number): void {
    this.#prepareExact(encoder);
    for (let s = 0; s < steps; s++) this.#encodeOps(encoder, this.#ops);
  }

  /** Record one pass over `ops` into `encoder`. */
  #encodeOps(encoder: GPUCommandEncoder, ops: Op[]): void {
    {
      let pass: GPUComputePassEncoder | null = null;
      const inPass = (): GPUComputePassEncoder => {
        if (!pass) pass = encoder.beginComputePass({ label: 'mgpu-step' });
        return pass;
      };
      const endPass = (): void => {
        if (pass) {
          pass.end();
          pass = null;
        }
      };
      for (const op of ops) {
        switch (op.kind) {
          case 'kernel': {
            const p = inPass();
            p.setPipeline(op.pipeline);
            p.setBindGroup(0, op.bindGroup);
            p.dispatchWorkgroups(Math.ceil(op.count / WORKGROUP_SIZE));
            if (op.copyBack) {
              endPass();
              encoder.copyBufferToBuffer(
                op.copyBack.from, 0, op.copyBack.to, 0, op.copyBack.bytes,
              );
            }
            break;
          }
          case 'synth':
            this.#shtInto(inPass(), op);
            break;
          case 'analys':
            this.#shtInto(inPass(), op);
            break;
          case 'dtheta':
            this.#derivInto(inPass(), op);
            break;
          case 'dphi':
            this.#derivInto(inPass(), op);
            break;
          case 'dthetac':
            this.#deriv!.encodeDthetacInto(inPass(), op.bindGroup);
            break;
          case 'dphic':
            this.#deriv!.encodeDphicInto(inPass(), op.bindGroup);
            break;
          case 'dphig':
            this.#sht.encodeDphigInto(inPass(), op.binding);
            break;
          case 'synth-batch':
            this.#sht.encodeSynthBatchInto(inPass(), op.binding);
            break;
          case 'analys-batch':
            this.#sht.encodeAnalysBatchInto(inPass(), op.binding);
            break;
          case 'dot': {
            const p = inPass();
            p.setPipeline(op.binding.pipeline);
            p.setBindGroup(0, op.binding.bindGroup);
            p.dispatchWorkgroups(1);
            break;
          }
          case 'lusolve':
            // #prepareExact guaranteed a site with current factors.
            op.call.site!.encodeSolve(inPass());
            break;
          case 'copy':
            endPass();
            encoder.copyBufferToBuffer(
              op.from, op.fromOffset ?? 0, op.to, op.toOffset ?? 0, op.bytes,
            );
            break;
        }
      }
      endPass();
    }
  }

  #shtInto(pass: GPUComputePassEncoder, op: Op & { kind: 'synth' | 'analys' }): void {
    if (op.kind === 'synth') this.#sht.encodeSynthInto(pass, op.binding);
    else this.#sht.encodeAnalysInto(pass, op.binding);
  }

  #derivInto(pass: GPUComputePassEncoder, op: Op & { kind: 'dtheta' | 'dphi' }): void {
    // planStatement already refused to plan a dtheta/dphi op without a
    // DerivPlan, so #deriv is guaranteed set whenever an op of this kind exists.
    if (op.kind === 'dtheta') this.#deriv!.encodeDthetaInto(pass, op.binding);
    else this.#deriv!.encodeDphiInto(pass, op.binding);
  }

  /**
   * Operations one run records, counting a `lusolve` by the dispatches its
   * block sweeps make rather than as one line — what the app's per-
   * submission budget wants to know. `describe()` still lists it once.
   */
  opCount(): number {
    let count = 0;
    for (const op of this.#ops) {
      count += op.kind === 'lusolve' ? (op.call.site?.solveDispatches ?? 1) : 1;
    }
    return count;
  }

  /**
   * Human-readable op sequence — what the .m actually compiled to. Batched
   * transforms list one line per lane, annotated: the line count equals the
   * logical op count regardless of the device's batch width, so op-count
   * assertions in the tests are batch-invariant.
   */
  describe(): string[] {
    return this.#ops.flatMap((op) => {
      if ('labels' in op) {
        const kind = op.kind === 'synth-batch' ? 'synth' : 'analys';
        return op.labels.map(
          (label, i) =>
            `${kind.padEnd(7)} ${label}  [batch lane ${i + 1}/${op.binding.size}]`,
        );
      }
      return [`${op.kind.padEnd(7)} ${op.label}`];
    });
  }

  destroy(): void {
    for (const b of this.#owned) b.destroy();
    this.#paramBuf.destroy();
    this.#owned.length = 0;
    for (const call of this.#exactCalls) {
      call.site?.destroy();
      call.site = null;
    }
  }
}

/**
 * `x = synth(y)` / `x = dot(y, z)` -> the call's name and arguments. A
 * buffer argument must be a plain variable (an expression would need its own
 * buffer, which is exactly what writing it on its own line provides — the
 * per-argument check is in the planner); an index argument may be any
 * expression the plan can evaluate (`j + 1`), and `randnfun3`'s wavelength
 * may be a literal or a model parameter.
 */
function externalCall(stmt: Assign): { name: string; args: IRExpr[] } | null {
  const e = stmt.expr;
  if (e.kind !== 'Call') return null;
  const arity = EXTERNAL_OPS.get(e.name);
  if (!arity) return null;
  if (e.args.length < arity.minArgs || e.args.length > arity.maxArgs) {
    const want =
      arity.minArgs === arity.maxArgs
        ? `${arity.minArgs}`
        : `${arity.minArgs} to ${arity.maxArgs}`;
    throw new UnsupportedOnGpu(
      `'${e.name}' takes ${want} argument${arity.maxArgs === 1 ? '' : 's'}`,
      stmt.span,
    );
  }
  return { name: e.name, args: e.args };
}

/** A `randnfun3` wavelength argument: a literal, or the parameter to read it
 *  from when the host fills the coefficient table. */
export type Randnfun3Lambda =
  | { kind: 'const'; value: number }
  | { kind: 'param'; name: string };

const sameLambda = (a: Randnfun3Lambda, b: Randnfun3Lambda): boolean =>
  a.kind === 'const' && b.kind === 'const'
    ? a.value === b.value
    : a.kind === 'param' && b.kind === 'param' && a.name === b.name;

/** The wavelength value a plan's `randnfun3` call resolves to. */
export const resolveLambda = (
  lambda: Randnfun3Lambda,
  params: Record<string, number>,
): number => (lambda.kind === 'const' ? lambda.value : params[lambda.name]);

const extArgName = (a: IRExpr): string =>
  a.kind === 'Var' ? a.name : a.kind === 'NumLit' ? String(a.value) : '<expression>';

function collectVars(
  e: IRExpr,
  visit: (v: Extract<IRExpr, { kind: 'Var' }>) => void,
): void {
  const walk = (x: IRExpr): void => {
    switch (x.kind) {
      case 'Var':
        visit(x);
        return;
      case 'Binary':
        walk(x.left);
        walk(x.right);
        return;
      case 'Unary':
        walk(x.operand);
        return;
      case 'Call':
        x.args.forEach(walk);
        return;
      default:
        return;
    }
  };
  walk(e);
}
