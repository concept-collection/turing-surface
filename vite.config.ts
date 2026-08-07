import { defineConfig } from 'vite';
import { realpathSync } from 'node:fs';
import { resolve } from 'node:path';

// numbl is a local `file:` dependency, so node_modules/numbl is a symlink to
// the sibling checkout. Its package `exports` map only publishes the runtime
// entry points, not the compiler internals we need (parser + JIT lowering), so
// we reach them through a path alias. (package.json's `imports` field cannot
// express this — Node rejects node_modules targets — and plain Node could not
// resolve numbl's internal `.js`->`.ts` imports anyway, which is why the GPU
// tests run in the browser harness rather than under `node`.)
// Realpath'd through the symlink: dev serves modules under their real ids, so
// aliasing the node_modules path would give the same file two identities (one
// per spelling) and run its side effects twice — the interpreter's builtin
// registry throws on the second.
const numblSrc = realpathSync(resolve(import.meta.dirname, 'node_modules/numbl/src'));

export default defineConfig({
  base: './',
  resolve: {
    alias: { 'numbl-src': numblSrc },
  },
  server: {
    // the alias resolves outside the project root (through the symlink)
    fs: { allow: [import.meta.dirname, numblSrc] },
  },
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        test: resolve(import.meta.dirname, 'test.html'),
      },
    },
  },
});
