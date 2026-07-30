/**
 * MATLAB source -> numbl's JIT IR, ready for the WGSL backend.
 *
 * A model file defines ordinary MATLAB functions; the host specializes the ones
 * it needs (`init`, `step`) for the concrete argument types of the current grid.
 * This is exactly how numbl drives its own JIT — the caller supplies argument
 * types, and lowering fixes every type and shape from there.
 *
 * Driving it through function signatures rather than injected scope means the
 * .m declares what it needs: each parameter name is matched against what the
 * host offers, and a name the host does not provide is a compile error rather
 * than a silently undefined variable.
 *
 * Two numbl passes matter here:
 *  - `specializeUserFunction` lowers one function to IR, one statement per
 *    operation (ANF), with every node's type fixed.
 *  - `inlinePass` then folds single-use temps back into their consumer, so a
 *    source line like `fu = a - u + u.*u.*v` becomes ONE statement whose RHS is
 *    an expression tree — i.e. one GPU kernel instead of four.
 */
import { parseMFile } from 'numbl-src/numbl-core/parser/index.ts';
import { Workspace, Lowerer, tensorDouble, scalarDouble } from 'numbl-src/numbl-core/jit/index.ts';
import { specializeUserFunction } from 'numbl-src/numbl-core/jit/lowering/specialize.ts';
import { inlinePass } from 'numbl-src/numbl-core/jit/codegen/inlinePass.ts';
import type { For, IRExpr, IRFunc, IRStmt } from 'numbl-src/numbl-core/jit/lowering/ir.ts';
import type { Type } from 'numbl-src/numbl-core/jit/lowering/types.ts';
import { externalOpFiles, type GridSizes } from './externals.ts';
import { ModelCompileError } from './errors.ts';

/** What the host can supply for an argument the .m declares. */
export type Binding =
  /** An array, passed in a GPU buffer. */
  | { kind: 'tensor'; shape: number[] }
  /** A tunable scalar. Deliberately carries no exact value: an exact scalar
   *  would be constant-folded into the kernels, so moving a slider would force
   *  a recompile instead of just rewriting a uniform. */
  | { kind: 'param' }
  /** A fixed scalar, exact so array constructors reading it keep static
   *  shapes. */
  | { kind: 'const'; value: number };

const typeOf = (b: Binding): Type => {
  switch (b.kind) {
    case 'tensor':
      return tensorDouble(b.shape);
    case 'param':
      return scalarDouble('unknown');
    case 'const':
      // Carry the sign too: numbl's sign lattice decides, for instance,
      // whether sqrt() of a value can go complex.
      return scalarDouble(
        b.value > 0 ? 'positive' : b.value < 0 ? 'negative' : 'zero',
        b.value,
      );
  }
};

/** One specialized function, as the planner consumes it. */
export interface CompiledFunction {
  name: string;
  /** Declared arguments, in order, with the cName each lowered to. */
  params: { name: string; cName: string; binding: Binding }[];
  /** Requested outputs, in order, with the cName holding each result. */
  outputs: { name: string; cName: string; ty: Type }[];
  /** The lowered body. Read this only after `finish()`: the inline pass
   *  REPLACES the statement array rather than mutating it, so this is a live
   *  view of the function rather than a snapshot. */
  readonly body: IRStmt[];
}

/** The shape of a `function` statement in numbl's AST. */
interface FunctionDecl {
  type: 'Function';
  name: string;
  params: string[];
  outputs: string[];
}

/**
 * A parsed model. Specialize the functions you need, then call `finish()` once
 * — the inline pass rewrites every specialization together.
 */
export class CompiledModel {
  #lowerer: Lowerer;
  #decls: Map<string, FunctionDecl>;
  #bindings: Record<string, Binding>;

  constructor(
    source: string,
    bindings: Record<string, Binding>,
    grid: GridSizes,
    fileName = 'model.m',
  ) {
    const ast = parseMFile(source, fileName);
    const ws = new Workspace(fileName, []);
    ws.addFile({ name: fileName, source, ast });
    // synth / analys become resolvable, with their type rules.
    for (const f of externalOpFiles(grid)) ws.addFile(f);
    ws.finalize();

    this.#bindings = bindings;
    this.#lowerer = new Lowerer(ws);
    this.#decls = new Map();
    for (const stmt of ast.body as { type: string }[]) {
      if (stmt.type === 'Function') {
        const fn = stmt as unknown as FunctionDecl;
        this.#decls.set(fn.name, fn);
      }
    }
  }

  /** Names of the functions the file defines. */
  functionNames(): string[] {
    return [...this.#decls.keys()];
  }

  /**
   * Lower `name` for the current bindings, requesting `nargout` outputs.
   * Every declared parameter must name something the host provides.
   */
  specialize(name: string, nargout: number): CompiledFunction {
    const decl = this.#decls.get(name);
    if (!decl) {
      const defined = this.functionNames();
      throw new ModelCompileError(
        `the model must define a function named '${name}'` +
          (defined.length
            ? ` (it defines ${defined.map((n) => `'${n}'`).join(', ')})`
            : ' (it defines no functions)'),
      );
    }
    if (decl.outputs.length < nargout) {
      throw new ModelCompileError(
        `'${name}' must return ${nargout} value${nargout === 1 ? '' : 's'}, ` +
          `but declares ${decl.outputs.length}`,
      );
    }

    const bindings = decl.params.map((p) => {
      const b = this.#bindings[p];
      if (!b) {
        const offered = Object.keys(this.#bindings).join(', ');
        throw new ModelCompileError(
          `'${name}' takes an argument named '${p}', which this app does not ` +
            `provide. Available: ${offered}.`,
        );
      }
      return b;
    });

    const fn: IRFunc = specializeUserFunction.call(
      this.#lowerer,
      decl,
      bindings.map(typeOf),
      undefined,
      undefined,
      undefined,
      nargout,
      undefined,
    );

    return {
      name,
      params: fn.params.map((p, i) => ({
        name: p,
        cName: fn.cParams[i],
        binding: bindings[i],
      })),
      outputs: fn.outputs.slice(0, nargout).map((o, i) => ({
        name: o,
        cName: fn.cOutputs[i],
        ty: fn.outputTypes[i],
      })),
      // A getter, not a snapshot: `finish()` runs after every specialization
      // and swaps in a rewritten statement array.
      get body() {
        return fn.body;
      },
    };
  }

  /**
   * Run the inline pass over everything specialized so far. It rewrites the
   * function bodies in place, so `CompiledFunction`s handed out earlier are
   * updated too.
   */
  finish(): void {
    // Snapshot what each loop body assigns, before the pass can rewrite it.
    const loops = [...this.#lowerer.specializations.values()].flatMap((fn) =>
      forLoops(fn.body).map((loop) => ({
        fn,
        loop,
        assignedBefore: assignedCNames(loop.body),
      })),
    );

    inlinePass({ topLevelStmts: [], functions: this.#lowerer.specializations });

    for (const { fn, loop, assignedBefore } of loops) checkLoopEscapes(fn, loop, assignedBefore);
  }
}

/** Every `for` in a statement list, including nested ones. */
function forLoops(stmts: IRStmt[]): For[] {
  const out: For[] = [];
  const walk = (list: IRStmt[]): void => {
    for (const s of list) {
      if (s.kind === 'For') {
        out.push(s);
        walk(s.body);
      }
    }
  };
  walk(stmts);
  return out;
}

/** cNames assigned anywhere in a statement list, including inside loops. */
function assignedCNames(stmts: IRStmt[]): Set<string> {
  const out = new Set<string>();
  const walk = (list: IRStmt[]): void => {
    for (const s of list) {
      if (s.kind === 'Assign') out.add(s.cName);
      else if (s.kind === 'For') walk(s.body);
    }
  };
  walk(stmts);
  return out;
}

/** Call `visit` for every variable read in an expression. */
function forEachVarRead(e: IRExpr, visit: (cName: string) => void): void {
  const walk = (x: IRExpr): void => {
    switch (x.kind) {
      case 'Var':
        return visit(x.cName);
      case 'Binary':
        walk(x.left);
        walk(x.right);
        return;
      case 'Unary':
        walk(x.operand);
        return;
      case 'Call':
        x.args.forEach(walk);
        return;
      default:
        return;
    }
  };
  walk(e);
}

/**
 * Refuse a loop whose result the inline pass folded away.
 *
 * numbl's inline pass substitutes a single-use producer into its consumer and
 * drops the producer. Inside a loop body it runs with no protected names — it
 * counts uses within that body alone — so an assignment whose only *visible*
 * use is later in the same body can be elided even though something outside
 * the loop still wants the value.
 *
 * That elision is correct for a body-local temp, which is what makes fusion
 * work inside the loop, and it is caught downstream in the two cases where the
 * value has no buffer at all: the planner already refuses a declared output
 * that is never assigned, and a read of a name it never allocated. The case it
 * would not catch is a variable assigned *before* the loop as well — there the
 * buffer exists, holding the pre-loop value, and the loop would silently
 * contribute nothing. So check all three here, in one place, against what the
 * body assigned before the pass ran.
 */
function checkLoopEscapes(fn: IRFunc, loop: For, assignedBefore: Set<string>): void {
  const assignedAfter = assignedCNames(loop.body);
  const elided = [...assignedBefore].filter((c) => !assignedAfter.has(c));
  if (elided.length === 0) return;

  // Reads anywhere in the function outside this loop's own body.
  const readOutside = new Set<string>();
  const walk = (list: IRStmt[]): void => {
    for (const s of list) {
      if (s === (loop as IRStmt)) continue; // the loop's own body is not "outside"
      if (s.kind === 'Assign') forEachVarRead(s.expr, (c) => readOutside.add(c));
      else if (s.kind === 'For') walk(s.body);
    }
  };
  walk(fn.body);

  const outputs = new Set(fn.cOutputs);
  const escaping = elided.filter((c) => readOutside.has(c) || outputs.has(c));
  if (escaping.length === 0) return;

  const names = [...new Set(escaping)].map((c) => `'${c}'`).join(', ');
  throw new ModelCompileError(
    `inside the 'for' loop, ${names} is assigned but only read later in the ` +
      `same iteration, so the compiler folded the assignment into its reader — ` +
      `yet the value is also wanted outside the loop. Read it once outside the ` +
      `loop instead, or use it more than once inside it.`,
  );
}
