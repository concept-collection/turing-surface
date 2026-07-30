/**
 * The available geometries: their MATLAB source, and the metadata the host owns.
 *
 * The same split as the model registry — the .m is the shape, everything around
 * it (parameter names, defaults, slider ranges) lives here, and the .m declares
 * which of them it wants by naming them as arguments.
 *
 * `sphere` is first and is not merely one entry among several: it is the case
 * the whole project is checked against, where the surface is exactly the unit
 * sphere and every result must match turing-sphere.
 */
import sphereSource from '../../geometries/sphere.m?raw';
import ellipsoidSource from '../../geometries/ellipsoid.m?raw';
import peanutSource from '../../geometries/peanut.m?raw';
import bumpySource from '../../geometries/bumpy.m?raw';
import type { ParamSpec, Params } from '../mgpu/registry.ts';

export interface MGeometry {
  key: string;
  label: string;
  blurb: string;
  params: ParamSpec[];
  /** MATLAB source — the shape itself. */
  source: string;
}

const sphere: MGeometry = {
  key: 'sphere',
  label: 'Sphere',
  blurb: 'The unit sphere. The reference case: identical to turing-sphere.',
  params: [],
  source: sphereSource,
};

const ellipsoid: MGeometry = {
  key: 'ellipsoid',
  label: 'Ellipsoid',
  blurb: 'Each axis scaled independently. Exactly degree 1, so nothing is lost to band-limiting.',
  params: [
    { key: 'ax', label: 'a', value: 1.5, min: 0.2, max: 3, step: 0.05 },
    { key: 'ay', label: 'b', value: 1, min: 0.2, max: 3, step: 0.05 },
    { key: 'az', label: 'c', value: 0.6, min: 0.2, max: 3, step: 0.05 },
  ],
  source: ellipsoidSource,
};

const peanut: MGeometry = {
  key: 'peanut',
  label: 'Peanut',
  blurb: 'A dumbbell pinched at the equator: two positively curved ends joined by a saddle.',
  params: [
    { key: 'waist', label: 'waist', value: 0.6, min: 0, max: 0.9, step: 0.05 },
    { key: 'stretch', label: 'stretch', value: 0.6, min: 0, max: 2, step: 0.05 },
  ],
  source: peanutSource,
};

const bumpy: MGeometry = {
  key: 'bumpy',
  label: 'Bumpy',
  blurb: 'Equatorial lobes plus a pear-shaped offset — curvature that varies in both angles.',
  params: [
    { key: 'amp', label: 'amp', value: 0.3, min: 0, max: 0.6, step: 0.02 },
    { key: 'nlobe', label: 'lobes', value: 5, min: 1, max: 12, step: 1 },
    { key: 'pear', label: 'pear', value: 0.15, min: -0.4, max: 0.4, step: 0.05 },
  ],
  source: bumpySource,
};

export const mGeometries: MGeometry[] = [sphere, ellipsoid, peanut, bumpy];

export const mGeometryByKey = (key: string): MGeometry | undefined =>
  mGeometries.find((g) => g.key === key);

export const defaultGeometryParams = (g: MGeometry): Params =>
  Object.fromEntries(g.params.map((p) => [p.key, p.value]));

/** The geometry every result is checked against, and what a caller who names
 *  none gets: the case where the solver is exact. */
export const SPHERE_KEY = 'sphere';

/** What the app and the benchmark start on. Not the sphere: this project
 *  exists for the other shapes, and opening on the reference case would hide
 *  the one thing it adds. */
export const DEFAULT_GEOMETRY_KEY = 'ellipsoid';
