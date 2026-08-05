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
import type { Assign, For, IRExpr, IRStmt } from 'numbl-src/numbl-core/jit/lowering/ir.ts';
import type { NumericType, Type } from 'numbl-src/numbl-core/jit/lowering/types.ts';
import { ShtPlan, type ShtBinding } from '../sht/sht.ts';
import { DerivPlan, type DerivBinding } from '../sht/deriv.ts';
import { ReducePlan, type DotBinding } from './reduce.ts';
import type { CompiledFunction } from './compile.ts';
import { EXTERNAL_OPS } from './externals.ts';
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
    }
  | { kind: 'synth' | 'analys'; binding: ShtBinding; label: string }
  | { kind: 'dtheta' | 'dphi'; binding: DerivBinding; label: string }
  | { kind: 'dot'; binding: DotBinding; label: string }
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

  #device: GPUDevice;
  #sht: ShtPlan;
  #deriv?: DerivPlan;
  #ops: Op[];
  #owned: GPUBuffer[];
  #paramBuf: GPUBuffer;
  #paramData: Float32Array;
  /** Public name -> buffer, for uploading initial state and reading results. */
  #byName: Map<string, Slot>;

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

    const ops: Op[] = [];
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
      ops.push({
        kind: 'copy',
        from: src.buffer,
        to: dst.buffer,
        bytes: 4 * src.count,
        label: `${out.name} -> ${to}`,
      });
    });

    return new ModelPlan({
      device, sht, deriv, ops, byName, owned, paramBuf, paramData, paramNames,
    });

    async function planStatement(stmt: IRStmt): Promise<void> {
      if (stmt.kind === 'ReturnFromFunction') return; // nothing follows it
      if (stmt.kind === 'For') return planFor(stmt);
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
          ops.push({
            kind: 'dot',
            binding: await reduce.createDotBinding(a.buffer, b.buffer, dest.buffer, a.count),
            label,
          });
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
            ops.push({
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
              ops.push({
                kind: 'copy', from: bank.buffer, to: dest.buffer,
                bytes: 4 * bank.count, label: `${label} (bank copy)`,
              });
            }
            ops.push({
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
            ops.push({
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
              ops.push({
                kind: 'copy', from: base.buffer, to: dest.buffer,
                bytes: 4 * base.count, label: `${label} (base copy)`,
              });
            }
            ops.push({
              kind: 'copy', from: value.buffer,
              to: dest.buffer, toOffset: 4 * offset, bytes: 4, label,
            });
          }
          return;
        }
        const src = argSlot(0);
        if (ext.name === 'synth') {
          ops.push({
            kind: 'synth',
            binding: sht.createSynthBinding(src.buffer, dest.buffer),
            label,
          });
        } else if (ext.name === 'analys') {
          ops.push({
            kind: 'analys',
            binding: sht.createAnalysBinding(src.buffer, dest.buffer),
            label,
          });
        } else if (ext.name === 'dtheta' || ext.name === 'dphi') {
          if (!deriv) {
            throw new UnsupportedOnGpu(
              `'${ext.name}' needs the surface's derivative transforms, ` +
                `which this plan was not given`,
              stmt.span,
            );
          }
          ops.push(
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

      ops.push({
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
   * Record `steps` timesteps. Synchronous: no awaits, no readback. All of the
   * ops share one compute pass, which WebGPU executes in submission order
   * with a barrier between dispatches.
   */
  encodeSteps(encoder: GPUCommandEncoder, steps: number): void {
    for (let s = 0; s < steps; s++) {
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
      for (const op of this.#ops) {
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
          case 'dot': {
            const p = inPass();
            p.setPipeline(op.binding.pipeline);
            p.setBindGroup(0, op.binding.bindGroup);
            p.dispatchWorkgroups(1);
            break;
          }
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

  /** Human-readable op sequence — what the .m actually compiled to. */
  describe(): string[] {
    return this.#ops.map((op) => `${op.kind.padEnd(7)} ${op.label}`);
  }

  destroy(): void {
    for (const b of this.#owned) b.destroy();
    this.#paramBuf.destroy();
    this.#owned.length = 0;
  }
}

/**
 * `x = synth(y)` / `x = dot(y, z)` -> the call's name and arguments. A
 * buffer argument must be a plain variable (an expression would need its own
 * buffer, which is exactly what writing it on its own line provides — the
 * per-argument check is in the planner); an index argument may be any
 * expression the plan can evaluate (`j + 1`).
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
