/**
 * The shared MATLAB utilities in `tools/`, as interpreter workspace files.
 *
 * A geometry or a seeding draw is evaluated by numbl's interpreter (see
 * src/geom/geometry.ts), which resolves a call like `randnfunsphere(...)`
 * against the workspace files it is handed. Everything in `tools/` is handed
 * to every such run, so any .m can call any tool by name — MATLAB's own path
 * semantics, where the file name is the function name.
 *
 * These are *not* available to the models: a model's step compiles to WGSL,
 * where none of this exists.
 */
const sources = import.meta.glob('../tools/*.m', {
  query: '?raw',
  eager: true,
  import: 'default',
}) as Record<string, string>;

export interface ToolFile {
  name: string;
  source: string;
}

/** Every tool, named as MATLAB wants it (`randnfunsphere.m`). */
export const toolFiles: ToolFile[] = Object.entries(sources)
  .map(([path, source]) => ({ name: path.slice(path.lastIndexOf('/') + 1), source }))
  .sort((a, b) => a.name.localeCompare(b.name));
