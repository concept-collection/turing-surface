/**
 * Expand calls to user-defined MATLAB functions into the caller's body.
 *
 * numbl lowers `dL = dlap(X, ...)` to a single `Call` statement whose cName
 * names the callee's specialization — one IRFunc per distinct argument-type
 * signature, shared across call sites. The WGSL planner, though, executes a
 * flat statement list: every value is a buffer, every statement a dispatch,
 * and a call boundary has no runtime meaning on the GPU. So this pass gives a
 * call the only meaning it can have there: the callee's lowered body, spliced
 * in at the call site.
 *
 * Each call site gets its own clone of the callee body, with every
 * callee-local cName made unique to the site — so two calls to the same
 * function get separate buffers, while the *same* site re-planned per
 * unrolled loop iteration keeps reusing its buffers (the planner keys buffers
 * by cName, and the clone is made once, here, not per iteration).
 *
 * Arguments bind by renaming, not copying, wherever that is sound: a callee
 * parameter whose argument is a plain variable — after the inline pass they
 * all are, since it never folds an expression into a call argument — reads
 * the caller's buffer directly, unless the callee reassigns the parameter, in
 * which case a copy is materialized so the caller's value is not clobbered.
 * Outputs are the same in reverse: assignments to a callee output become
 * assignments to the caller's target variable, which is what makes a
 * loop-carried output (a solver iterating its result) work unchanged.
 *
 * The pass runs before numbl's inline (fusion) pass, which works per function
 * and treats a user call as an opaque producer. With every call already
 * spliced away, fusion sees one flat body and folds it exactly as it would
 * the same code written out by hand — a callee's final assignment can fuse
 * into its consumer, and the compiled plan is identical either way.
 */
import type {
  Assign,
  For,
  IRExpr,
  IRFunc,
  IRStmt,
  Span,
} from 'numbl-src/numbl-core/jit/lowering/ir.ts';
import { ModelCompileError } from './errors.ts';

/** A located compile failure. A span from a shared lib file still carries its
 *  offsets; they are only ever mapped onto the model source for display, so a
 *  failure inside a lib mislocates but still names the construct. */
const fail = (message: string, span: Span): ModelCompileError =>
  new ModelCompileError(message, { start: span.start, end: span.end });

/** Looks up a callee's specialization by its mangled cName. */
export type ResolveFn = (cName: string) => IRFunc | undefined;

/**
 * Expand every user-function call in `fn`, recursively, mutating `fn.body`.
 * Returns true if anything was expanded.
 */
export function expandUserCalls(fn: IRFunc, resolve: ResolveFn): boolean {
  const ctx = { resolve, site: 0, expanded: false };
  fn.body = expandBody(fn.body, ctx, [fn.cName]);
  return ctx.expanded;
}

interface Ctx {
  resolve: ResolveFn;
  /** Call-site counter, for unique cNames. */
  site: number;
  expanded: boolean;
}

function expandBody(stmts: IRStmt[], ctx: Ctx, stack: string[]): IRStmt[] {
  const out: IRStmt[] = [];
  for (const stmt of stmts) {
    if (stmt.kind === 'Assign' && stmt.expr.kind === 'Call') {
      const callee = ctx.resolve(stmt.expr.cName);
      if (callee) {
        out.push(
          ...expandCall(
            callee,
            stmt.expr.args,
            [{ name: stmt.name, cName: stmt.cName }],
            ctx,
            stack,
            stmt,
          ),
        );
        continue;
      }
    }
    if (stmt.kind === 'MultiAssignCall' && !stmt.isBuiltin) {
      const callee = ctx.resolve(stmt.cName);
      if (!callee) {
        throw fail(
          `call to '${stmt.name}' does not resolve to a function in this model`,
          stmt.span,
        );
      }
      out.push(
        ...expandCall(
          callee,
          stmt.args,
          stmt.outputs.map((o) => o.binding),
          ctx,
          stack,
          stmt,
        ),
      );
      continue;
    }
    if (stmt.kind === 'For') {
      stmt.body = expandBody(stmt.body, ctx, stack);
      out.push(stmt);
      continue;
    }
    out.push(stmt);
  }
  return out;
}

/**
 * One call site: the callee's body, cloned and renamed into the caller's
 * namespace, preceded by whatever argument bindings need materializing.
 */
function expandCall(
  callee: IRFunc,
  args: IRExpr[],
  outs: ({ name: string; cName: string } | null)[],
  ctx: Ctx,
  stack: string[],
  at: IRStmt,
): IRStmt[] {
  if (stack.includes(callee.cName)) {
    throw fail(
      `'${callee.name}' calls itself (perhaps through another function); ` +
        `recursion cannot be compiled to a fixed sequence of GPU operations`,
      at.span,
    );
  }
  ctx.expanded = true;
  const site = ++ctx.site;
  /** Site-unique cName for a callee-local. */
  const local = (cName: string): string => `${callee.name}$${site}$${cName}`;
  /** Display name for a callee-local — shows up in buffer labels and in
   *  describe(), so a solver's internals read as `richardson#1.X`. */
  const display = (name: string): string => `${callee.name}#${site}.${name}`;

  const assigned = assignedCNames(callee.body);
  const rename = new Map<string, string>();
  const names = new Map<string, string>();
  const prelude: IRStmt[] = [];

  // Outputs first: assignments to a callee output become assignments to the
  // caller's target. An ignored output (`~`, or trailing outputs the caller
  // did not ask for) stays a site-local.
  callee.cOutputs.forEach((c, j) => {
    const target = j < outs.length ? outs[j] : null;
    if (target) {
      rename.set(c, target.cName);
      names.set(c, target.name);
    } else {
      rename.set(c, local(c));
      names.set(c, display(callee.outputs[j]));
    }
  });

  // Parameters: rename onto the argument where sound, else materialize.
  callee.cParams.forEach((p, i) => {
    const arg = args[i];
    if (arg === undefined) {
      throw fail(
        `'${callee.name}' takes ${callee.cParams.length} arguments, ` +
          `but this call passes ${args.length}`,
        at.span,
      );
    }
    if (rename.has(p)) {
      // The parameter is also an output (`function X = f(X)`): seed the
      // caller's target with the argument, and let the body update it there.
      prelude.push(makeAssign(names.get(p)!, rename.get(p)!, arg, callee.paramTypes[i], at));
    } else if (arg.kind === 'Var' && !assigned.has(p)) {
      rename.set(p, arg.cName);
      names.set(p, arg.name);
    } else {
      // A non-variable argument, or a parameter the callee reassigns: bind it
      // to a site-local first. For a scalar this is a free derived-scalar
      // binding; for a tensor it is one copy kernel.
      const c = local(p);
      rename.set(p, c);
      names.set(p, display(callee.params[i]));
      prelude.push(makeAssign(names.get(p)!, c, arg, callee.paramTypes[i], at));
    }
  });

  const body = cloneBody(callee.body, { rename, names, local, display, callee, at });
  return [...prelude, ...expandBody(body, ctx, [...stack, callee.cName])];
}

const makeAssign = (
  name: string,
  cName: string,
  expr: IRExpr,
  ty: IRFunc['paramTypes'][number],
  at: IRStmt,
): Assign => ({ kind: 'Assign', name, cName, ty, expr, span: at.span });

interface CloneCtx {
  /** Callee cName -> caller-namespace cName. Filled for outputs and params up
   *  front; locals are added on first sight. */
  rename: Map<string, string>;
  names: Map<string, string>;
  local: (cName: string) => string;
  display: (name: string) => string;
  callee: IRFunc;
  at: IRStmt;
}

function cloneBody(stmts: IRStmt[], c: CloneCtx): IRStmt[] {
  const out: IRStmt[] = [];
  for (const s of stmts) {
    switch (s.kind) {
      case 'Assign':
        out.push({
          ...s,
          name: mapName(s.name, s.cName, c),
          cName: mapCName(s.cName, s.name, c),
          expr: cloneExpr(s.expr, c),
        });
        break;
      case 'MultiAssignCall':
        out.push({
          ...s,
          args: s.args.map((a) => cloneExpr(a, c)),
          outputs: s.outputs.map((o) =>
            o.binding
              ? {
                  ...o,
                  binding: {
                    name: mapName(o.binding.name, o.binding.cName, c),
                    cName: mapCName(o.binding.cName, o.binding.name, c),
                  },
                }
              : o,
          ),
        });
        break;
      case 'For': {
        const loop: For = {
          ...s,
          cVar: mapCName(s.cVar, s.varName, c),
          start: cloneExpr(s.start, c),
          end: cloneExpr(s.end, c),
          body: [],
        };
        loop.body = cloneBody(s.body, c);
        out.push(loop);
        break;
      }
      case 'ReturnFromFunction':
        // The callee's return has no meaning at the splice point; statements
        // never follow it (numbl refuses an early return at lowering).
        break;
      default:
        throw fail(
          `'${c.callee.name}' contains a '${s.kind}' statement, which cannot ` +
            `be compiled to a fixed sequence of GPU operations`,
          s.span,
        );
    }
  }
  return out;
}

/** Caller-namespace cName for a callee-side cName, minting one for a local
 *  seen for the first time. */
function mapCName(cName: string, name: string, c: CloneCtx): string {
  const existing = c.rename.get(cName);
  if (existing) return existing;
  const fresh = c.local(cName);
  c.rename.set(cName, fresh);
  c.names.set(cName, c.display(name));
  return fresh;
}

function mapName(name: string, cName: string, c: CloneCtx): string {
  return c.names.get(cName) ?? c.display(name);
}

function cloneExpr(e: IRExpr, c: CloneCtx): IRExpr {
  switch (e.kind) {
    case 'Var':
      return {
        ...e,
        name: mapName(e.name, e.cName, c),
        cName: mapCName(e.cName, e.name, c),
      };
    case 'Binary':
      return { ...e, left: cloneExpr(e.left, c), right: cloneExpr(e.right, c) };
    case 'Unary':
      return { ...e, operand: cloneExpr(e.operand, c) };
    case 'Call':
      return { ...e, args: e.args.map((a) => cloneExpr(a, c)) };
    default:
      // Literals and anything else without variable reads: share as-is (the
      // planner treats expressions as read-only).
      return e;
  }
}

/** cNames assigned anywhere in a statement list, including loop variables. */
function assignedCNames(stmts: IRStmt[]): Set<string> {
  const out = new Set<string>();
  const walk = (list: IRStmt[]): void => {
    for (const s of list) {
      if (s.kind === 'Assign') out.add(s.cName);
      else if (s.kind === 'MultiAssignCall') {
        for (const o of s.outputs) if (o.binding) out.add(o.binding.cName);
      } else if (s.kind === 'For') {
        out.add(s.cVar);
        walk(s.body);
      }
    }
  };
  walk(stmts);
  return out;
}
