/**
 * Compile failures, reported in coordinates of the model file the user edits.
 *
 * Failures arrive from three places, each with its own idea of position:
 * numbl's parser (a `position` offset), numbl's lowerer (`UnsupportedConstruct`
 * / `JitTypeError`, with a `span`), and this project's WGSL emitter
 * (`UnsupportedOnGpu`, carrying the numbl span it was given). All of them are
 * offsets into the whole model file — the file is parsed once, and each function
 * is specialized from that one AST — so they need only be turned into a line and
 * column for the editor.
 */

/** A compile failure located in the full model source. */
export class ModelCompileError extends Error {
  /** Offset into the whole .m file, when the failure has a position. */
  readonly start?: number;
  readonly end?: number;
  /** Name of the model function being compiled. */
  readonly fn?: string;

  constructor(
    message: string,
    opts: { start?: number; end?: number; fn?: string; cause?: unknown } = {},
  ) {
    super(message, { cause: opts.cause });
    this.name = 'ModelCompileError';
    this.start = opts.start;
    this.end = opts.end;
    this.fn = opts.fn;
  }
}

/** Extract whatever position information an error carries. */
function positionOf(e: unknown): { start?: number; end?: number } {
  const span = (e as { span?: { start?: unknown; end?: unknown } }).span;
  if (span && typeof span.start === 'number') {
    return {
      start: span.start,
      end: typeof span.end === 'number' ? span.end : undefined,
    };
  }
  // numbl's parser SyntaxError reports a bare offset.
  const position = (e as { position?: unknown }).position;
  if (typeof position === 'number') return { start: position };
  return {};
}

/** Normalize any thrown value into a located `ModelCompileError`. */
function asCompileError(e: unknown, fn?: string): ModelCompileError {
  if (e instanceof ModelCompileError) return e;
  const { start, end } = positionOf(e);
  const raw = e instanceof Error ? e.message : String(e);
  // numbl's parse errors read as bare token complaints out of context.
  const message =
    (e as Error)?.name === 'SyntaxError' ? `MATLAB syntax error: ${raw}` : raw;
  return new ModelCompileError(message, { fn, start, end, cause: e });
}

/**
 * Run `fn`, locating any compile failure in the model file. Use for whole-file
 * phases (parsing) that belong to no single function.
 */
export function inModel<T>(fn: () => T): T {
  try {
    return fn();
  } catch (e) {
    throw asCompileError(e);
  }
}

/** Run `fn`, attributing any compile failure to the model function `name`. */
export function inFunction<T>(name: string, fn: () => T): T {
  try {
    return fn();
  } catch (e) {
    throw asCompileError(e, name);
  }
}

/** Async form of `inFunction`. */
export async function inFunctionAsync<T>(
  name: string,
  fn: () => Promise<T>,
): Promise<T> {
  try {
    return await fn();
  } catch (e) {
    throw asCompileError(e, name);
  }
}

/** Render a failure for display: message, section, and 1-based line/column. */
export function formatFailure(e: unknown, source: string): string {
  const message = e instanceof Error ? e.message : String(e);
  if (!(e instanceof ModelCompileError)) return message;
  const where: string[] = [];
  if (e.start !== undefined && e.start <= source.length) {
    const before = source.slice(0, e.start);
    const line = before.split('\n').length;
    const column = e.start - before.lastIndexOf('\n');
    where.push(`line ${line}, column ${column}`);
  }
  if (e.fn) where.push(`in ${e.fn}()`);
  return where.length ? `${message}  (${where.join(', ')})` : message;
}
