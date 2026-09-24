/**
 * `dot(x, y)` — the one reduction the solvers need, as a GPU operation.
 *
 * A Krylov solver's scalars (rho, alpha, omega, ...) are inner products of the
 * spectral state, and the whole point of the plan architecture is that no
 * value crosses back to the CPU mid-step — so the dot product must produce a
 * GPU-resident scalar: a 1-element buffer that later kernels read (the
 * planner binds any single-element value as `in<slot>[0]`, see
 * src/mgpu/wgsl.ts).
 *
 * One workgroup does the whole reduction: each thread accumulates a strided
 * partial sum, then a shared-memory tree combines them and thread 0 writes
 * the result. A single dispatch, no multi-pass bookkeeping, and — because
 * the striding is fixed — a bit-deterministic summation order on a given
 * device. n up to a few hundred thousand is a short loop per thread, far
 * below anything the solver grids produce.
 */

const WG = 256;

/** One dot call site: its bind group and the pipeline for its length. */
export interface DotBinding {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
}

const dotWGSL = (n: number): string => `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;

var<workgroup> partials: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var s = 0.0;
  var i = lid.x;
  loop {
    if (i >= ${n}u) { break; }
    s = s + a[i] * b[i];
    i = i + ${WG}u;
  }
  partials[lid.x] = s;
  workgroupBarrier();
  var stride = ${WG / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      partials[lid.x] = partials[lid.x] + partials[lid.x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lid.x == 0u) {
    out[0] = partials[0];
  }
}
`;

export class ReducePlan {
  #device: GPUDevice;
  #layout: GPUBindGroupLayout;
  /** Element count is baked into the shader, so pipelines cache per length. */
  #byN = new Map<number, GPUComputePipeline>();

  constructor(device: GPUDevice) {
    this.#device = device;
    const entry = (
      binding: number,
      type: GPUBufferBindingType,
    ): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    });
    this.#layout = device.createBindGroupLayout({
      entries: [entry(0, 'storage'), entry(1, 'read-only-storage'), entry(2, 'read-only-storage')],
    });
  }

  async #pipeline(n: number): Promise<GPUComputePipeline> {
    const cached = this.#byN.get(n);
    if (cached) return cached;
    const device = this.#device;
    device.pushErrorScope('validation');
    const code = dotWGSL(n);
    const module = device.createShaderModule({ code, label: `dot-${n}` });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) {
      throw new Error(
        `WGSL compile error in dot(${n}):\n` +
          errors.map((e) => `  ${e.lineNum}:${e.linePos} ${e.message}`).join('\n'),
      );
    }
    const pipeline = await device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.#layout] }),
      compute: { module, entryPoint: 'main' },
      label: `dot-${n}`,
    });
    const err = await device.popErrorScope();
    if (err) throw new Error(`pipeline dot(${n}): ${err.message}`);
    this.#byN.set(n, pipeline);
    return pipeline;
  }

  async createDotBinding(
    a: GPUBuffer,
    b: GPUBuffer,
    out: GPUBuffer,
    n: number,
  ): Promise<DotBinding> {
    const pipeline = await this.#pipeline(n);
    const bindGroup = this.#device.createBindGroup({
      layout: this.#layout,
      entries: [
        { binding: 0, resource: { buffer: out } },
        { binding: 1, resource: { buffer: a } },
        { binding: 2, resource: { buffer: b } },
      ],
    });
    return { pipeline, bindGroup };
  }

  encodeDotInto(pass: GPUComputePassEncoder, binding: DotBinding): void {
    pass.setPipeline(binding.pipeline);
    pass.setBindGroup(0, binding.bindGroup);
    pass.dispatchWorkgroups(1);
  }
}
