/** Vite's `?raw` suffix imports a file's text. Used to load .m model sources. */
declare module '*?raw' {
  const source: string;
  export default source;
}

/** Vite's `import.meta.glob`, used to load every .m in tools/ at once
 *  (src/tools.ts). Only the eager + `?raw` form this project uses is
 *  declared — it returns each match's text, keyed by path. */
interface ImportMeta {
  glob(
    pattern: string,
    options: { query: '?raw'; eager: true; import: 'default' },
  ): Record<string, string>;
}
