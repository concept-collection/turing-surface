/**
 * Correctness of the .m -> WGSL path, against closed-form answers.
 *
 * These replace what used to be a comparison against a second TypeScript
 * implementation of the same scheme. Checking against arithmetic is stronger:
 * two implementations agreeing only shows they share assumptions, whereas an
 * exact recurrence pins the result. Each test picks a case whose evolution is
 * known in closed form, runs it through the real pipeline — MATLAB source,
 * numbl lowering, generated WGSL, GPU transforms — and compares.
 *
 * A: a linear reaction makes every spherical-harmonic mode independent, with a
 *    known growth factor per degree. Checks the transform round-trip, the
 *    eigenvalue mapping, the IMEX update and the state feedback.
 * B: a nonlinear reaction on a uniform field follows the scalar ODE map
 *    exactly. Checks that a generated kernel evaluates a nonlinear reaction.
 * C: a small perturbation of the Schnakenberg fixed point follows the
 *    linearized 2x2 IMEX recurrence, and the expected mode is unstable. Checks
 *    a real two-species model.
 *
 * Everything runs in fp32 on the GPU, so tolerances are set by fp32 round-off
 * (~1e-7 relative) rather than by the scheme.
 */
import { ShtPlan } from '../src/sht/sht.ts';
import { gridForLmax, lmIndex, nlmCalc, type ShtConfig } from '../src/sht/layout.ts';
import { GpuModel } from '../src/mgpu/model.ts';
import { mModelByKey, defaultParams, type MModel, type ParamSpec } from '../src/mgpu/registry.ts';
import { Geometry } from '../src/geom/geometry.ts';
import { mGeometryByKey, SPHERE_KEY } from '../src/geom/registry.ts';
import linearSource from './models/linear.m?raw';
import logisticSource from './models/logistic.m?raw';

export type Check = (name: string, ok: boolean, detail: string) => void;
export type Log = (s: string) => void;

const param = (key: string, value: number): ParamSpec => ({
  key, label: key, value, min: -1e9, max: 1e9, step: 1,
});

/** A one-species test model with an arbitrary parameter list. */
const testModel = (key: string, source: string, params: string[]): MModel => ({
  key,
  label: key,
  blurb: '',
  species: ['u'],
  state: ['U'],
  params: params.map((p) => param(p, 0)),
  pdeg: 1,
  seedAmp: 1,
  source,
});

/**
 * Every closed-form case here is a statement about the *round sphere*, so
 * every model here is built on the sphere geometry. The models that take a
 * surface still get one — a geometry is always supplied, and for the sphere it
 * is the degree-1 embedding, which is what makes these answers exact.
 */
async function makeModel(
  device: GPUDevice,
  model: MModel,
  cfg: ShtConfig,
  niter = 1,
): Promise<{ sht: ShtPlan; gpu: GpuModel }> {
  const sht = await ShtPlan.create(device, cfg);
  const geometry = await Geometry.create({
    device,
    sht,
    cfg,
    source: mGeometryByKey(SPHERE_KEY)!.source,
    paramNames: [],
    params: {},
  });
  const gpu = await GpuModel.create({
    device,
    sht,
    cfg,
    source: model.source,
    paramNames: model.params.map((p) => p.key),
    state: model.state,
    view: model.species,
    geometry,
    niter,
  });
  return { sht, gpu };
}

export async function analyticChecks(
  device: GPUDevice,
  check: Check,
  log: Log,
): Promise<void> {
  // ---- A: linear reaction, exact per-mode growth factor -----------------
  {
    const lmax = 15;
    const { nlat, nphi } = gridForLmax(lmax, 1);
    const cfg = { lmax, mmax: lmax, nlat, nphi };
    const nlm = nlmCalc(lmax, lmax);
    const c = -0.3;
    const D = 0.01;
    const dt = 0.1;
    const nsteps = 20;

    const model = testModel('linear', linearSource, ['c', 'D', 'dt']);
    const { sht, gpu } = await makeModel(device, model, cfg);
    gpu.setParams({ c, D, dt });

    // A single (l, m) mode, written straight into the spectral state.
    const l = 5;
    const m = 2;
    const idx = lmIndex(lmax, l, m);
    const U0 = new Float32Array(2 * nlm);
    U0[2 * idx] = 0.8;
    U0[2 * idx + 1] = -0.35;
    gpu.upload('U', U0);

    gpu.step(nsteps);
    const U = await gpu.read('U');

    const g = (1 + dt * c) / (1 + dt * D * l * (l + 1));
    const factor = g ** nsteps;
    const wantRe = 0.8 * factor;
    const wantIm = -0.35 * factor;
    const errRe = Math.abs(U[2 * idx] - wantRe);
    const errIm = Math.abs(U[2 * idx + 1] - wantIm);
    check(
      'A: linear reaction follows the exact per-mode recurrence',
      errRe < 2e-6 && errIm < 2e-6,
      `err (${errRe.toExponential(2)}, ${errIm.toExponential(2)}) after ${nsteps} steps`,
    );

    // Nothing may leak into the other modes.
    let leak = 0;
    for (let i = 0; i < nlm; i++) {
      if (i === idx) continue;
      leak = Math.max(leak, Math.abs(U[2 * i]), Math.abs(U[2 * i + 1]));
    }
    check('A: no leakage into other modes', leak < 2e-6, `max |other| ${leak.toExponential(2)}`);

    gpu.destroy();
    sht.destroy();
  }

  // ---- B: nonlinear reaction on a uniform field, exact ODE map ----------
  {
    const lmax = 15;
    const { nlat, nphi } = gridForLmax(lmax, 3);
    const cfg = { lmax, mmax: lmax, nlat, nphi };
    const npts = nlat * nphi;
    const r = 0.7;
    const D = 0.01;
    const dt = 0.05;
    const nsteps = 25;
    const u0 = 0.3;

    const model = testModel('logistic', logisticSource, ['r', 'D', 'dt']);
    const { sht, gpu } = await makeModel(device, model, cfg);
    gpu.setParams({ r, D, dt });

    // Uniform initial field: stays uniform, and diffusion cannot touch it.
    const field = new Float32Array(npts).fill(u0);
    gpu.init(field);
    const Ustart = await gpu.read('U');
    gpu.step(nsteps);
    const Uend = await gpu.read('U');

    // Read the *state*, not the `u` output: a model computes its grid fields
    // from the state at the START of the step (`u = synth(U)` precedes the
    // update), so the rendered field lags the state by one step. The l=0
    // coefficient of a uniform field scales linearly with its value, so the
    // ratio gives the value back without needing Y_00's normalization.
    const got = u0 * (Uend[0] / Ustart[0]);

    let want = u0;
    for (let s = 0; s < nsteps; s++) want += dt * r * want * (1 - want);

    const err = Math.abs(got - want);
    check(
      'B: uniform nonlinear reaction follows the scalar ODE map',
      err < 5e-6,
      `${got.toFixed(7)} vs ${want.toFixed(7)}, err ${err.toExponential(2)}`,
    );

    // And it must still be uniform: any structure would mean the kernel is
    // reading the wrong elements.
    const u = await gpu.read('u');
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of u) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    check(
      'B: the field stays uniform',
      hi - lo < 1e-6,
      `spread ${(hi - lo).toExponential(2)}`,
    );

    gpu.destroy();
    sht.destroy();
  }

  // ---- C: linearized Turing recurrence on a real two-species model ------
  {
    const model = mModelByKey('schnakenberg')!;
    const p = defaultParams(model);
    const lmax = 31;
    const { nlat, nphi } = gridForLmax(lmax, model.pdeg);
    const cfg = { lmax, mmax: lmax, nlat, nphi };
    const nlm = nlmCalc(lmax, lmax);
    const npts = nlat * nphi;

    const { sht, gpu } = await makeModel(device, model, cfg);
    gpu.setParams(p);

    // Seed the exact homogeneous fixed point by handing init a zero
    // perturbation, then add a small single-mode bump to u only.
    gpu.init(new Float32Array(npts));
    const l = 24;
    const m = 7;
    const idx = lmIndex(lmax, l, m);
    const eps = 1e-6;
    const U0 = await gpu.read('U');
    const V0 = await gpu.read('V');
    const Upert = Float32Array.from(U0);
    Upert[2 * idx] += eps;
    gpu.upload('U', Upert);
    gpu.upload('V', V0);

    const nsteps = 40;
    gpu.step(nsteps);
    const U = await gpu.read('U');
    const V = await gpu.read('V');

    // Jacobian of (a - u + u^2 v, b - u^2 v) at the fixed point us = a+b,
    // vs = b/us^2, with diffusion applied implicitly per species.
    const us = p.a + p.b;
    const vs = p.b / (us * us);
    const J = [
      [-1 + 2 * us * vs, us * us],
      [-2 * us * vs, -us * us],
    ];
    const lam = l * (l + 1);
    const du = 1 / (1 + p.dt * p.D1 * lam);
    const dv = 1 / (1 + p.dt * p.D2 * lam);
    let cu = eps;
    let cv = 0;
    for (let s = 0; s < nsteps; s++) {
      const nu = (cu + p.dt * (J[0][0] * cu + J[0][1] * cv)) * du;
      const nv = (cv + p.dt * (J[1][0] * cu + J[1][1] * cv)) * dv;
      cu = nu;
      cv = nv;
    }

    const gotU = U[2 * idx] - U0[2 * idx];
    const gotV = V[2 * idx] - V0[2 * idx];
    const relU = Math.abs(gotU - cu) / Math.max(Math.abs(cu), 1e-30);
    const relV = Math.abs(gotV - cv) / Math.max(Math.abs(cv), 1e-30);
    // Looser than A and B by design: a 1e-6 perturbation sits on a state of
    // order 1, so fp32 keeps only ~4 significant digits of it.
    check(
      'C: perturbation follows the linearized 2x2 IMEX recurrence',
      relU < 5e-3 && relV < 5e-3,
      `rel err (${relU.toExponential(2)}, ${relV.toExponential(2)})`,
    );
    check(
      `C: the (l=${l}, m=${m}) mode is unstable`,
      Math.abs(cu) > eps && Math.abs(gotU) > eps,
      `|c_u| ${eps.toExponential(2)} -> ${Math.abs(gotU).toExponential(2)}`,
    );

    log(`  C: growth over ${nsteps} steps = ${(Math.abs(cu) / eps).toFixed(3)}x (predicted)`);

    gpu.destroy();
    sht.destroy();
  }
}
