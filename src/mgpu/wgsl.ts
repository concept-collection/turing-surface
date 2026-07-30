/**
 * IR expression tree -> one WGSL compute kernel.
 *
 * This is the WebGPU counterpart of numbl's C-side fused emitter
 * (`codegen/emitTensorFused.ts`): for an `Assign` whose right-hand side is
 * purely element-wise over operands of the target's shape, emit a single
 * kernel that computes one output element per invocation. Because numbl's
 * inline pass has already folded the ANF temps back together, one source line
 * of MATLAB becomes one kernel.
 *
 * Everything is f32, matching the existing fp32 WebGPU transform backend.
 */
import { getBuiltin } from 'numbl-src/numbl-core/jit/builtins/index.ts';
import { isMultiElement } from 'numbl-src/numbl-core/jit/lowering/types.ts';
import type { IRExpr, Assign } from 'numbl-src/numbl-core/jit/lowering/ir.ts';
import type { NumericType, Type } from 'numbl-src/numbl-core/jit/lowering/types.ts';

/** Raised for a construct the WGSL backend cannot express. Mirrors numbl's
 *  own decline discipline: fail at compile time with a source span, never
 *  silently produce something that computes the wrong thing. */
export class UnsupportedOnGpu extends Error {
  readonly span?: unknown;
  constructor(message: string, span?: unknown) {
    super(message);
    this.name = 'UnsupportedOnGpu';
    this.span = span;
  }
}

const isNumeric = (t: Type): t is NumericType => t.kind === 'Numeric';
const isTensor = (t: Type): boolean => isNumeric(t) && isMultiElement(t);

/** Element-wise binary builtins -> WGSL infix operator. */
const BINARY_OPS: Record<string, string> = {
  plus: '+',
  minus: '-',
  times: '*',
  rdivide: '/',
  // Degenerate to element-wise when at least one side is a scalar; the
  // both-tensor (true matrix) case is rejected below.
  mtimes: '*',
  mrdivide: '/',
};

/** Element-wise unary builtins -> WGSL prefix operator. */
const UNARY_OPS: Record<string, string> = { uminus: '-', uplus: '+' };

/** Element-wise builtin calls -> WGSL builtin of the same arity. */
const CALL_FNS: Record<string, string> = {
  abs: 'abs',
  acos: 'acos',
  asin: 'asin',
  atan: 'atan',
  atan2: 'atan2',
  ceil: 'ceil',
  cos: 'cos',
  cosh: 'cosh',
  exp: 'exp',
  floor: 'floor',
  log: 'log',
  log2: 'log2',
  max: 'max',
  min: 'min',
  round: 'round',
  sign: 'sign',
  sin: 'sin',
  sinh: 'sinh',
  sqrt: 'sqrt',
  tan: 'tan',
  tanh: 'tanh',
};

/** WGSL f32 literal. Must always carry a decimal point or exponent, or WGSL
 *  infers AbstractInt and rejects the mixed-type arithmetic. */
function f32Lit(v: number): string {
  if (!Number.isFinite(v)) {
    throw new UnsupportedOnGpu(`cannot emit non-finite literal ${v}`);
  }
  return Number.isInteger(v) && Math.abs(v) < 1e21
    ? `${v}.0`
    : String(v).includes('e')
      ? `${v}f`
      : String(v);
}

/** How a scalar or tensor operand is read inside the kernel. */
export interface KernelInputs {
  /** cName -> storage binding index, for multi-element tensor operands. */
  tensors: Map<string, number>;
  /** cName -> slot in the params storage buffer, for runtime scalars. */
  params: Map<string, number>;
  /** cName -> defining expression, for scalars the .m computes from
   *  parameters (`us = a + b`). These have no buffer and no param slot; they
   *  become `let` bindings in the prologue of every kernel that reads them. */
  scalars: Map<string, { name: string; expr: IRExpr }>;
}

/** Mutable state while emitting one kernel. */
interface Ctx {
  io: KernelInputs;
  /** `let` lines to emit before the body, in dependency order. */
  prologue: string[];
  /** cName -> WGSL identifier, for scalars already bound in the prologue. */
  bound: Map<string, string>;
}

/** WGSL identifier for a derived scalar. Avoids a leading underscore, which
 *  WGSL reserves. */
const scalarIdent = (cName: string): string =>
  `s_${cName.replace(/[^A-Za-z0-9_]/g, '_')}`;

/**
 * Bind a .m-derived scalar in the prologue (once), after whatever it depends
 * on, and return its identifier.
 */
function bindScalar(cName: string, ctx: Ctx): string {
  const already = ctx.bound.get(cName);
  if (already) return already;
  const def = ctx.io.scalars.get(cName)!;
  const ident = scalarIdent(cName);
  // Claim the name before emitting the RHS so a (malformed) self-reference
  // cannot recurse forever.
  ctx.bound.set(cName, ident);
  const rhs = emitExpr(def.expr, ctx);
  ctx.prologue.push(`  let ${ident} = ${rhs};`);
  return ident;
}

/**
 * Emit the per-element WGSL expression for `e`. `i` is the element index
 * variable in scope.
 */
function emitExpr(e: IRExpr, ctx: Ctx): string {
  const io = ctx.io;
  switch (e.kind) {
    case 'NumLit':
      return f32Lit(e.value);

    case 'Var': {
      if (isTensor(e.ty)) {
        const slot = io.tensors.get(e.cName);
        if (slot === undefined) {
          throw new UnsupportedOnGpu(`no buffer bound for '${e.name}'`, e.span);
        }
        return `in${slot}[i]`;
      }
      // Scalar: either an exact compile-time value or a runtime parameter.
      if (isNumeric(e.ty) && typeof e.ty.exact === 'number') {
        return f32Lit(e.ty.exact);
      }
      const slot = io.params.get(e.cName);
      if (slot !== undefined) return `prm[${slot}]`;
      if (io.scalars.has(e.cName)) return bindScalar(e.cName, ctx);
      throw new UnsupportedOnGpu(
        `scalar '${e.name}' is not a constant, a parameter, or computed in ` +
          `this model`,
        e.span,
      );
    }

    case 'Binary': {
      if ((e.builtin === 'mtimes' || e.builtin === 'mrdivide') &&
          isTensor(e.left.ty) && isTensor(e.right.ty)) {
        throw new UnsupportedOnGpu(
          `matrix '${e.builtin === 'mtimes' ? '*' : '/'}' is not supported; ` +
            `use the element-wise form ('.${e.builtin === 'mtimes' ? '*' : '/'}')`,
          e.span,
        );
      }
      if (e.builtin === 'power' || e.builtin === 'mpower') {
        return emitPower(e.left, e.right, ctx, e.span);
      }
      const op = BINARY_OPS[e.builtin];
      if (!op) {
        throw new UnsupportedOnGpu(`operator '${e.builtin}' is not supported`, e.span);
      }
      return `(${emitExpr(e.left, ctx)} ${op} ${emitExpr(e.right, ctx)})`;
    }

    case 'Unary': {
      const op = UNARY_OPS[e.builtin];
      if (!op) {
        throw new UnsupportedOnGpu(`unary '${e.builtin}' is not supported`, e.span);
      }
      return `(${op}${emitExpr(e.operand, ctx)})`;
    }

    case 'Call': {
      // A shape constructor used inside an element-wise expression
      // contributes the same constant at every slot, so it needs no buffer.
      // (The shape itself is validated against the target by checkShapes.)
      if (e.name === 'ones') return '1.0';
      if (e.name === 'zeros') return '0.0';

      const fn = CALL_FNS[e.name];
      const b = getBuiltin(e.name);
      if (!fn || !b?.elementwise) {
        // A call numbl resolved to another function in the file gets a mangled
        // specialization name; a builtin keeps its source-level name. Only the
        // model's entry points are compiled, so a helper is a distinct failure
        // from an unsupported builtin and deserves to say so.
        const isUserFunction = e.cName !== e.name;
        throw new UnsupportedOnGpu(
          isUserFunction
            ? `'${e.name}' is a function defined in this model. Only init and ` +
                `step are compiled — inline its body into the caller.`
            : `'${e.name}' cannot be evaluated element-wise on the GPU`,
          e.span,
        );
      }
      return `${fn}(${e.args.map((a) => emitExpr(a, ctx)).join(', ')})`;
    }

    default:
      throw new UnsupportedOnGpu(`'${e.kind}' is not supported on the GPU`, e.span);
  }
}

/**
 * `x.^k`. WGSL's `pow` is undefined for a negative base, and these fields go
 * negative routinely, so expand small non-negative integer exponents into
 * repeated multiplication — which is also what makes `u.^2` free.
 */
function emitPower(base: IRExpr, exponent: IRExpr, ctx: Ctx, span: unknown): string {
  const k =
    exponent.kind === 'NumLit'
      ? exponent.value
      : isNumeric(exponent.ty) && typeof exponent.ty.exact === 'number'
        ? exponent.ty.exact
        : undefined;
  const b = emitExpr(base, ctx);
  if (k !== undefined && Number.isInteger(k) && k >= 0 && k <= 8) {
    if (k === 0) return '1.0';
    // bind once so a compound base expression is not re-evaluated k times
    return `pow_i${k}(${b})`;
  }
  if (k !== undefined && Number.isInteger(k) && k < 0 && k >= -8) {
    return `(1.0 / pow_i${-k}(${b}))`;
  }
  throw new UnsupportedOnGpu(
    `'.^' needs a literal integer exponent in [-8, 8] (got ` +
      `${k === undefined ? 'a runtime value' : k}); a negative base makes ` +
      `WGSL's pow() undefined`,
    span,
  );
}

/** Fixed-exponent power helpers, emitted only when used. */
function powHelpers(used: Set<number>): string {
  const out: string[] = [];
  for (const k of [...used].sort((a, b) => a - b)) {
    const body =
      k === 1 ? 'x' : `x${' * x'.repeat(k - 1)}`;
    out.push(`fn pow_i${k}(x: f32) -> f32 { return ${body}; }`);
  }
  return out.join('\n');
}

/**
 * Reject implicit expansion (broadcasting).
 *
 * numbl's lowering permits it — `2x4096 .* 1x4096` lowers happily with MATLAB
 * expansion semantics — but a kernel that walks one linear index across every
 * operand would quietly compute the wrong thing. So every multi-element
 * operand must have exactly the target's shape. Scalars are fine: they are
 * read from the params buffer or folded in as literals.
 */
function checkShapes(e: IRExpr, target: NumericType, name: string): void {
  const want = target.shape;
  const same = (t: NumericType): boolean => {
    const got = t.shape;
    return (
      !!want && !!got && want.length === got.length &&
      want.every((d, i) => d === got[i])
    );
  };
  const walk = (x: IRExpr): void => {
    if (isNumeric(x.ty) && isMultiElement(x.ty) && !same(x.ty)) {
      const got = x.ty.shape?.join('x') ?? 'dynamic';
      throw new UnsupportedOnGpu(
        `'${name}' would need implicit expansion: an operand is ${got} but the ` +
          `result is ${want?.join('x') ?? 'dynamic'}. Expand it explicitly ` +
          `(the GPU kernel walks one index across every operand).`,
        x.span,
      );
    }
    switch (x.kind) {
      case 'Binary':
        walk(x.left);
        walk(x.right);
        return;
      case 'Unary':
        walk(x.operand);
        return;
      case 'Call':
        // A shape constructor's own arguments are sizes, not data.
        if (x.name !== 'ones' && x.name !== 'zeros') x.args.forEach(walk);
        return;
      default:
        return;
    }
  };
  walk(e);
}

export const WORKGROUP_SIZE = 64;

export interface Kernel {
  code: string;
  /** Number of output elements. */
  count: number;
  label: string;
}

/**
 * Build the kernel for one element-wise `Assign`. `io` must already map every
 * tensor operand cName to a binding index and every runtime scalar to a
 * params slot; the output is binding 0 and the params buffer is the binding
 * after the last input.
 */
export function buildKernel(
  stmt: Assign,
  io: KernelInputs,
  count: number,
  label: string,
): Kernel {
  if (!isNumeric(stmt.ty)) {
    throw new UnsupportedOnGpu(`'${stmt.name}' is not a numeric array`, stmt.span);
  }
  if (stmt.ty.isComplex) {
    throw new UnsupportedOnGpu(
      `'${stmt.name}' is complex; the GPU backend is real-only (a spectral ` +
        `field is carried as a real 2 x nlm array)`,
      stmt.span,
    );
  }

  checkShapes(stmt.expr, stmt.ty, stmt.name);
  const ctx: Ctx = { io, prologue: [], bound: new Map() };
  const body = emitExpr(stmt.expr, ctx);

  // pow_iK helpers are discovered during emission; scan the result for them.
  const used = new Set<number>();
  const emitted = [...ctx.prologue, body].join('\n');
  for (const m of emitted.matchAll(/\bpow_i(\d+)\(/g)) used.add(Number(m[1]));

  const decls = [`@group(0) @binding(0) var<storage, read_write> out: array<f32>;`];
  for (const [, slot] of io.tensors) {
    decls.push(
      `@group(0) @binding(${slot + 1}) var<storage, read> in${slot}: array<f32>;`,
    );
  }
  // Params live in a read-only storage buffer rather than a uniform block:
  // uniform arrays would need 16-byte element stride.
  const prmBinding = io.tensors.size + 1;
  decls.push(
    `@group(0) @binding(${prmBinding}) var<storage, read> prm: array<f32>;`,
  );

  const code = `${decls.join('\n')}

${powHelpers(used)}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
${ctx.prologue.length ? `${ctx.prologue.join('\n')}\n` : ''}  out[i] = ${body};
}
`;
  return { code, count, label };
}
