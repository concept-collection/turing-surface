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

/**
 * The compile-time value of a scalar expression, if it has one. A literal
 * carries its own; a variable carries one when it was bound to a `const` (the
 * host's fixed scalars) or computed from constants, because numbl propagates
 * `exact` through the type lattice.
 */
const exactValue = (e: IRExpr): number | undefined => {
  if (isNumeric(e.ty) && typeof e.ty.exact === 'number') return e.ty.exact;
  return e.kind === 'NumLit' ? e.value : undefined;
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
  | { kind: 'copy'; from: GPUBuffer; to: GPUBuffer; bytes: number; label: string };

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
  #ops: Op[];
  #owned: GPUBuffer[];
  #paramBuf: GPUBuffer;
  #paramData: Float32Array;
  /** Public name -> buffer, for uploading initial state and reading results. */
  #byName: Map<string, Slot>;

  private constructor(init: {
    device: GPUDevice;
    sht: ShtPlan;
    ops: Op[];
    byName: Map<string, Slot>;
    owned: GPUBuffer[];
    paramBuf: GPUBuffer;
    paramData: Float32Array;
    paramNames: string[];
  }) {
    this.#device = init.device;
    this.#sht = init.sht;
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
      device, sht, ops, byName, owned, paramBuf, paramData, paramNames,
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
      if (!isTensor(stmt.ty)) {
        // A scalar the model derives from its parameters (`us = a + b`). It
        // gets no buffer and no dispatch: the kernels that read it bind it as
        // a `let` in their prologue.
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

      const ext = externalCall(stmt);
      if (ext) {
        const argSlot = slots.get(ext.argCName);
        if (!argSlot) {
          throw new UnsupportedOnGpu(
            `'${ext.name}' reads '${ext.argName}', which has no buffer`,
            stmt.span,
          );
        }
        ops.push(
          ext.name === 'synth'
            ? {
                kind: 'synth',
                binding: sht.createSynthBinding(argSlot.buffer, dest.buffer),
                label: `${stmt.name} = synth(${ext.argName})`,
              }
            : {
                kind: 'analys',
                binding: sht.createAnalysBinding(argSlot.buffer, dest.buffer),
                label: `${stmt.name} = analys(${ext.argName})`,
              },
        );
        return;
      }

      // Element-wise kernel. Collect the distinct tensor operands and give
      // them dense binding slots.
      const tensors = new Map<string, number>();
      collectTensorVars(stmt.expr, (cName) => {
        if (!tensors.has(cName)) tensors.set(cName, tensors.size);
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
      const from = exactValue(stmt.start);
      const to = exactValue(stmt.end);
      if (from === undefined || to === undefined) {
        throw new UnsupportedOnGpu(
          `a 'for' loop is unrolled into the op sequence, so its bounds must ` +
            `be known when the model is compiled — ` +
            `${from === undefined ? 'the start' : 'the end'} of this one is a ` +
            `runtime value. Use a whole number, or a count the app supplies ` +
            `as a fixed argument (changing it recompiles).`,
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
          case 'copy':
            endPass();
            encoder.copyBufferToBuffer(op.from, 0, op.to, 0, op.bytes);
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

/** `x = synth(y)` / `x = analys(y)` -> the call's name and argument. */
function externalCall(
  stmt: Assign,
): { name: string; argCName: string; argName: string } | null {
  const e = stmt.expr;
  if (e.kind !== 'Call' || !EXTERNAL_OPS.has(e.name)) return null;
  if (e.args.length !== 1 || e.args[0].kind !== 'Var') {
    throw new UnsupportedOnGpu(
      `'${e.name}' must be applied to a single variable`,
      stmt.span,
    );
  }
  const arg = e.args[0];
  return { name: e.name, argCName: arg.cName, argName: arg.name };
}

function collectTensorVars(e: IRExpr, visit: (cName: string) => void): void {
  const walk = (x: IRExpr): void => {
    switch (x.kind) {
      case 'Var':
        if (isTensor(x.ty)) visit(x.cName);
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
