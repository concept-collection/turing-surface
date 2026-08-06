/**
 * Drawing a seed field's Fourier modes, off the main thread.
 *
 * The draw is `tools/randnfun3.m` in numbl's interpreter, and its cost goes as
 * the inverse cube of the wavelength: milliseconds at the default lambda, but
 * ~13 s at lambda 0.01. Synchronous JS that long does not merely feel slow —
 * it blocks the event loop outright, so the page stops painting and the
 * browser offers to kill it. Nothing about the draw needs the main thread
 * (it touches no GPU and no DOM), so it runs here and the result is
 * transferred back.
 */
import { drawModes, type BoundingBox } from './randnfun3.ts';

export interface DrawRequest {
  id: number;
  lambda: number;
  box: BoundingBox;
  seed: number;
  npts: number;
}

export type DrawReply =
  | { id: number; table: Float32Array; error?: undefined }
  | { id: number; table?: undefined; error: string };

self.onmessage = (e: MessageEvent<DrawRequest>): void => {
  const { id, lambda, box, seed, npts } = e.data;
  let reply: DrawReply;
  let transfer: Transferable[] = [];
  try {
    const table = drawModes(lambda, box, seed, npts);
    reply = { id, table };
    transfer = [table.buffer];
  } catch (err) {
    reply = { id, error: err instanceof Error ? err.message : String(err) };
  }
  (self as unknown as {
    postMessage: (m: DrawReply, t: Transferable[]) => void;
  }).postMessage(reply, transfer);
};
