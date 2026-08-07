/**
 * Reading a reference .h5 in the page.
 *
 * h5wasm's browser build carries the whole HDF5 library as embedded wasm —
 * about 4 MB — so it is imported here, dynamically, and nowhere else: the page
 * pays for it on the first file actually loaded, never on startup. The bytes
 * are written into the wasm module's in-memory filesystem under a fixed
 * scratch name (loads are sequential — there is one file input), opened,
 * extracted, and unlinked.
 */
import { extractReferenceCase, type H5Node, type ReferenceCase } from './referenceCase.ts';

const SCRATCH = '/loaded-reference.h5';

export async function loadReferenceFile(file: File): Promise<ReferenceCase> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const h5 = await import('h5wasm');
  const { FS } = (await h5.ready) as unknown as {
    FS: { writeFile(path: string, data: Uint8Array): void; unlink(path: string): void };
  };
  FS.writeFile(SCRATCH, bytes);
  const opened = new h5.File(SCRATCH, 'r');
  try {
    return extractReferenceCase(opened as unknown as H5Node, file.name);
  } finally {
    opened.close();
    FS.unlink(SCRATCH);
  }
}
