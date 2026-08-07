/**
 * The reference-file reader, against a file this test writes itself.
 *
 * No GPU: this is about the format — that what h5wasm writes in the
 * documented layout (docs/ellipsoid-reference-spec.md) comes back through
 * `extractReferenceCase` with nothing renamed, rescaled or truncated, and
 * that a file the replay could not act on is refused with a message rather
 * than half-read. The h5wasm module is injected: the node harness passes
 * `h5wasm/node` (real files), the browser harness `h5wasm` (in-memory wasm
 * filesystem) — so the browser run also proves the wasm build actually ships.
 */
import { extractReferenceCase, type H5Node } from '../src/compare/referenceCase.ts';
import { nlmCalc } from '../src/sht/layout.ts';

type Check = (name: string, ok: boolean, detail: string) => void;
type Log = (line: string) => void;

/** The slice of h5wasm's writing API these checks touch — the node and
 *  browser builds both satisfy it structurally. */
interface H5Out {
  create_group(name: string): H5Out;
  create_attribute(name: string, data: unknown): void;
  create_dataset(args: { name: string; data: unknown; dtype?: string }): unknown;
}
export interface H5Rt {
  ready: Promise<unknown>;
  File: new (path: string, mode?: string) => H5Out & H5Node & { close(): unknown };
}

const LMAX = 3;
const STEPS = 8;

export async function referenceChecks(
  h5: H5Rt,
  /** Where a named scratch file may live: a temp dir on node, '/' in the
   *  browser's in-memory filesystem. */
  pathFor: (name: string) => string,
  check: Check,
  log: Log,
): Promise<void> {
  log('\nreference files (HDF5 layout):');
  const mod = (await h5.ready) as { FS?: { unlink(path: string): void } };
  const nlm = nlmCalc(LMAX, LMAX);
  const series = (offset: number): Float32Array =>
    Float32Array.from({ length: 2 * nlm }, (_, i) => offset + i / 16);
  const arrays = {
    Gx: series(100), Gy: series(200), Gz: series(300),
    initialU: series(1), finalU: series(2),
  };

  // ---- write the documented layout, read it back ---------------------------
  const goodPath = pathFor('ref-roundtrip.h5');
  {
    const f = new h5.File(goodPath, 'w');
    f.create_attribute('model', 'allencahn');
    f.create_attribute('species', ['U']);
    const spec = f.create_group('spec');
    spec.create_attribute('geometry', 'ellipsoid');
    spec.create_attribute('lmax', LMAX);
    spec.create_attribute('steps', STEPS);
    spec.create_attribute('niter', 2);
    spec.create_attribute('seed', 1);
    spec.create_attribute('warmup', 0);
    const params = spec.create_group('params');
    params.create_attribute('dt', 0.0625);
    params.create_attribute('eps2', 0.5);
    const geomParams = spec.create_group('geometry_params');
    geomParams.create_attribute('ax', 2.5);
    geomParams.create_attribute('ay', 1.25);
    geomParams.create_attribute('az', 0.75);
    const geom = f.create_group('geometry');
    geom.create_dataset({ name: 'Gx', data: arrays.Gx, dtype: '<f4' });
    geom.create_dataset({ name: 'Gy', data: arrays.Gy, dtype: '<f4' });
    geom.create_dataset({ name: 'Gz', data: arrays.Gz, dtype: '<f4' });
    f.create_group('initial').create_dataset({ name: 'U', data: arrays.initialU, dtype: '<f4' });
    f.create_group('final').create_dataset({ name: 'U', data: arrays.finalU, dtype: '<f4' });
    f.close();
  }
  {
    const f = new h5.File(goodPath, 'r');
    const rc = extractReferenceCase(f, 'ref-roundtrip.h5');
    f.close();
    mod.FS?.unlink(goodPath);

    check(
      'reference: the run identity survives the round trip',
      rc.model.key === 'allencahn' && rc.geometry.key === 'ellipsoid' &&
        rc.lmax === LMAX && rc.steps === STEPS && rc.niter === 2,
      `${rc.model.key} on ${rc.geometry.key}, lmax ${rc.lmax}, ` +
        `${rc.steps} steps, niter ${rc.niter}`,
    );
    check(
      'reference: the file’s parameters override the defaults',
      rc.params.dt === 0.0625 && rc.params.eps2 === 0.5 &&
        rc.geometryParams.ax === 2.5 && rc.geometryParams.ay === 1.25 &&
        rc.geometryParams.az === 0.75,
      `dt ${rc.params.dt}, eps2 ${rc.params.eps2}, ` +
        `ax/ay/az ${rc.geometryParams.ax}/${rc.geometryParams.ay}/${rc.geometryParams.az}`,
    );
    const same = (a: Float32Array, b: Float32Array): boolean =>
      a.length === b.length && a.every((v, i) => v === b[i]);
    check(
      'reference: every coefficient array comes back bit-exact',
      same(rc.geometryCoeffs.X, arrays.Gx) && same(rc.geometryCoeffs.Y, arrays.Gy) &&
        same(rc.geometryCoeffs.Z, arrays.Gz) && same(rc.initial.U, arrays.initialU) &&
        same(rc.final.U, arrays.finalU),
      `5 arrays x ${2 * nlm} float32 values`,
    );
  }

  // ---- a file the replay cannot act on is refused, not half-read -----------
  {
    const badPath = pathFor('ref-unknown-model.h5');
    const f = new h5.File(badPath, 'w');
    f.create_attribute('model', 'nosuchmodel');
    f.close();
    const r = new h5.File(badPath, 'r');
    let message = '';
    try {
      extractReferenceCase(r, 'ref-unknown-model.h5');
    } catch (e) {
      message = e instanceof Error ? e.message : String(e);
    }
    r.close();
    mod.FS?.unlink(badPath);
    check(
      'reference: an unknown model is refused with its name',
      message.includes('nosuchmodel'),
      message || 'no error thrown',
    );
  }
}
