/** Vite's `?raw` suffix imports a file's text. Used to load .m model sources. */
declare module '*?raw' {
  const source: string;
  export default source;
}
