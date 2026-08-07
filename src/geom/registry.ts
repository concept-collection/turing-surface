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
import blobSource from '../../geometries/blob.m?raw';
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
  blurb: 'The unit sphere — the reference case.',
  params: [],
  source: sphereSource,
};

const ellipsoid: MGeometry = {
  key: 'ellipsoid',
  label: 'Ellipsoid',
  blurb: 'The sphere with each axis scaled independently.',
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
  blurb: 'A dumbbell pinched at the equator.',
  params: [
    { key: 'waist', label: 'waist', value: 0.6, min: 0, max: 0.9, step: 0.05 },
    { key: 'stretch', label: 'stretch', value: 0.6, min: 0, max: 2, step: 0.05 },
  ],
  source: peanutSource,
};

const bumpy: MGeometry = {
  key: 'bumpy',
  label: 'Bumpy',
  blurb: 'Equatorial lobes plus a pear-shaped offset.',
  params: [
    { key: 'amp', label: 'amp', value: 0.3, min: 0, max: 0.6, step: 0.02 },
    { key: 'nlobe', label: 'lobes', value: 5, min: 1, max: 12, step: 1 },
    { key: 'pear', label: 'pear', value: 0.15, min: -0.4, max: 0.4, step: 0.05 },
  ],
  source: bumpySource,
};

const blob: MGeometry = {
  key: 'blob',
  label: 'Blob',
  blurb: 'The sphere warped by a smooth random function — a fresh shape per seed.',
  params: [
    { key: 'amp', label: 'amp', value: 0.5, min: 0, max: 0.8, step: 0.05 },
    { key: 'scale', label: 'λ', value: 1, min: 0.5, max: 3, step: 0.1 },
    { key: 'seed', label: 'seed', value: 1, min: 0, max: 9999, step: 1, reseed: true },
  ],
  source: blobSource,
};

export const mGeometries: MGeometry[] = [sphere, ellipsoid, peanut, bumpy, blob];

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
