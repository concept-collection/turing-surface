/**
 * The available models: their MATLAB source, and the metadata the host owns.
 *
 * A model's *algorithm* lives in its .m file. Everything around it lives here:
 * the parameter names the .m may take as arguments, their defaults and slider
 * ranges, which grid fields to render, and the dealiasing degree. The .m
 * declares nothing about these — it just names the parameters it wants, and
 * `CompiledModel` matches each against this table.
 *
 * Naming convention, documented in each .m:
 *   `u`, `v`, ...  grid fields the model computes and the app renders
 *   `U`, `V`, ...  the corresponding spectral state (uppercase)
 */
import schnakenbergSource from '../../models/schnakenberg.m?raw';
import brusselatorSource from '../../models/brusselator.m?raw';
import allencahnSource from '../../models/allencahn.m?raw';

export type Params = Record<string, number>;

/** A tunable scalar the .m may take as an argument. */
export interface ParamSpec {
  key: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
}

export interface MModel {
  key: string;
  label: string;
  blurb: string;
  /** Grid fields to render, one panel each. */
  species: string[];
  /** Spectral state names the .m advances. */
  state: string[];
  params: ParamSpec[];
  /** Polynomial degree of the reaction in the fields, for grid dealiasing. */
  pdeg: number;
  /** Amplitude of the seeded perturbation handed to `init`. */
  seedAmp: number;
  /** MATLAB source — the algorithm itself. */
  source: string;
}

/** Spectral state names follow the grid-field names, uppercased. */
const stateFor = (species: string[]): string[] => species.map((s) => s.toUpperCase());

const schnakenberg: MModel = {
  key: 'schnakenberg',
  label: 'Schnakenberg',
  blurb:
    'Turing spots. The homogeneous state is stable to uniform perturbations ' +
    'but unstable to degrees 14 ≤ l ≤ 40, most strongly at l = 24.',
  species: ['u', 'v'],
  state: stateFor(['u', 'v']),
  params: [
    { key: 'a', label: 'a', value: 0.1, min: 0.01, max: 0.5, step: 0.01 },
    { key: 'b', label: 'b', value: 0.9, min: 0.1, max: 2, step: 0.05 },
    { key: 'D1', label: 'D₁', value: 4e-4, min: 1e-5, max: 5e-3, step: 1e-5 },
    { key: 'D2', label: 'D₂', value: 8e-3, min: 1e-4, max: 5e-2, step: 1e-4 },
    { key: 'dt', label: 'dt', value: 0.05, min: 0.005, max: 0.5, step: 0.005 },
  ],
  pdeg: 3,
  seedAmp: 1e-2,
  source: schnakenbergSource,
};

const brusselator: MModel = {
  key: 'brusselator',
  label: 'Brusselator',
  blurb:
    'Turing stripes and spots, from a smaller diffusivity contrast than ' +
    'Schnakenberg but with a stiffer reaction.',
  species: ['u', 'v'],
  state: stateFor(['u', 'v']),
  params: [
    { key: 'A', label: 'A', value: 3, min: 0.5, max: 6, step: 0.1 },
    { key: 'B', label: 'B', value: 9, min: 1, max: 15, step: 0.25 },
    { key: 'D1', label: 'D₁', value: 3.33e-3, min: 1e-4, max: 2e-2, step: 1e-4 },
    { key: 'D2', label: 'D₂', value: 1.67e-2, min: 1e-3, max: 1e-1, step: 1e-3 },
    { key: 'dt', label: 'dt', value: 0.02, min: 0.002, max: 0.1, step: 0.002 },
  ],
  pdeg: 3,
  seedAmp: 1e-2,
  source: brusselatorSource,
};

const allencahn: MModel = {
  key: 'allencahn',
  label: 'Allen–Cahn',
  blurb:
    'A single species: interfaces form and then coarsen until one domain ' +
    'swallows the sphere.',
  species: ['u'],
  state: stateFor(['u']),
  params: [
    { key: 'eps2', label: 'ε²', value: 1e-3, min: 1e-4, max: 1e-2, step: 1e-4 },
    { key: 'dt', label: 'dt', value: 0.02, min: 0.002, max: 0.2, step: 0.002 },
  ],
  pdeg: 3,
  seedAmp: 1e-2,
  source: allencahnSource,
};

export const mModels: MModel[] = [schnakenberg, brusselator, allencahn];

export const mModelByKey = (key: string): MModel | undefined =>
  mModels.find((m) => m.key === key);

export const defaultParams = (m: MModel): Params =>
  Object.fromEntries(m.params.map((p) => [p.key, p.value]));

/** Named parameter presets shown in the UI dropdown. The pattern length scale
 *  goes as 1/sqrt(D), so scaling both diffusivities moves the spot size without
 *  changing the dynamics. */
export interface Preset {
  key: string;
  label: string;
  modelKey: string;
  /** Overrides applied on top of the model's default parameters. */
  params?: Params;
}

export const presets: Preset[] = [
  { key: 'schnak-spots', label: 'Schnakenberg — spots', modelKey: 'schnakenberg' },
  {
    key: 'schnak-coarse',
    label: 'Schnakenberg — coarse spots',
    modelKey: 'schnakenberg',
    params: { D1: 1e-3, D2: 2e-2 },
  },
  {
    key: 'schnak-fine',
    label: 'Schnakenberg — fine spots',
    modelKey: 'schnakenberg',
    params: { D1: 1.6e-4, D2: 3.2e-3 },
  },
  { key: 'brussel', label: 'Brusselator — stripes & spots', modelKey: 'brusselator' },
  { key: 'allencahn', label: 'Allen–Cahn — coarsening', modelKey: 'allencahn' },
];
