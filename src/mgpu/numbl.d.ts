/**
 * The numbl compiler surface this project depends on.
 *
 * We reach past numbl's published entry points into its JIT internals (parser,
 * lowerer, IR, inline pass), which its package `exports` map does not expose.
 * Those imports resolve through the `numbl-src` alias in vite.config.ts; these
 * declarations are what TypeScript checks against.
 *
 * Declaring the surface here rather than type-checking numbl's sources
 * directly keeps this project's compiler settings independent of numbl's, and
 * pins the exact contract we rely on. If numbl changes one of these shapes,
 * the build breaks here with a clear diff rather than deep inside its tree.
 *
 * Only the nodes the WGSL backend actually walks are spelled out; every other
 * IR kind is collapsed into a catch-all so that unhandled constructs are
 * rejected with a message instead of being silently mis-compiled.
 */

declare module 'numbl-src/numbl-core/jit/lowering/types.ts' {
  export type Sign =
    | 'positive' | 'nonneg' | 'negative' | 'nonpositive'
    | 'zero' | 'nonzero' | 'unknown';

  export type DimInfo = { kind: 'exact'; value: number } | { kind: 'unknown' };

  export type NumericExact =
    | number
    | Float64Array
    | { re: number; im: number }
    | { re: Float64Array; im: Float64Array };

  export interface NumericType {
    kind: 'Numeric';
    elem: 'double' | 'logical' | 'char' | string;
    isComplex: boolean;
    dims: DimInfo[];
    /** Present iff every dim is exact. */
    shape?: number[];
    sign: Sign;
    exact?: NumericExact;
  }

  /** Everything the WGSL backend rejects. */
  export interface NonNumericType {
    kind: 'Void' | 'Unknown' | 'String' | 'Handle' | 'Struct' | 'Class' | 'Cell';
  }

  export type Type = NumericType | NonNumericType;

  export function isMultiElement(t: NumericType): boolean;
  export function tensorDouble(shape: number[], exact?: Float64Array): NumericType;
  export function scalarDouble(sign?: Sign, exact?: number): NumericType;
}

declare module 'numbl-src/numbl-core/jit/lowering/ir.ts' {
  import type { Type } from 'numbl-src/numbl-core/jit/lowering/types.ts';

  export interface Span {
    file: string;
    start: number;
    end: number;
  }

  export interface NumLit {
    kind: 'NumLit';
    value: number;
    ty: Type;
    span: Span;
  }
  export interface Var {
    kind: 'Var';
    name: string;
    cName: string;
    ty: Type;
    span: Span;
  }
  export interface Binary {
    kind: 'Binary';
    builtin: string;
    left: IRExpr;
    right: IRExpr;
    ty: Type;
    span: Span;
  }
  export interface Unary {
    kind: 'Unary';
    builtin: string;
    operand: IRExpr;
    ty: Type;
    span: Span;
  }
  export interface Call {
    kind: 'Call';
    cName: string;
    name: string;
    args: IRExpr[];
    ty: Type;
    span: Span;
  }
  /** Any other IR expression kind — rejected by the WGSL emitter. */
  export interface OtherExpr {
    kind:
      | 'ImagLit' | 'StringLit' | 'TensorBuild' | 'TensorConcat' | 'CellLit'
      | 'CellEmpty' | 'CellIndexLoad' | 'HandleLit' | 'HandleCaptureLoad'
      | 'StructLit' | 'MemberLoad' | 'IndexLoad' | 'IndexSlice' | 'EndRef'
      | 'MakeRange';
    ty: Type;
    span: Span;
  }

  export type IRExpr = NumLit | Var | Binary | Unary | Call | OtherExpr;

  export interface Assign {
    kind: 'Assign';
    name: string;
    cName: string;
    ty: Type;
    expr: IRExpr;
    span: Span;
  }
  /**
   * A counted loop. The planner unrolls it, so only the fields that decide
   * the trip count and the loop variable's value are spelled out. `step` is
   * already a literal number in the IR — numbl rejects a non-literal step
   * during lowering — while `start` and `end` are expressions that must carry
   * an exact value for the planner to accept the loop.
   */
  export interface For {
    kind: 'For';
    /** Loop variable, as written in the .m. */
    varName: string;
    /** Loop variable's cName, the key the planner binds its value under. */
    cVar: string;
    start: IRExpr;
    step: number;
    end: IRExpr;
    body: IRStmt[];
    span: Span;
  }
  /**
   * A multi-output call statement, `[a, b] = f(...)`. For a user function
   * (`isBuiltin` false or absent) `cName` is the callee's specialization and
   * the call is expanded into the caller (src/mgpu/inlineCalls.ts); a
   * multi-output builtin is rejected by the planner.
   */
  export interface MultiAssignCall {
    kind: 'MultiAssignCall';
    /** Mangled specialization cName (user function) or builtin name. */
    cName: string;
    /** Source-level callee name, for diagnostics. */
    name: string;
    isBuiltin?: boolean;
    args: IRExpr[];
    /** One entry per output slot; `binding` null for an ignored output. */
    outputs: ReadonlyArray<{
      ty: Type;
      binding: { name: string; cName: string } | null;
    }>;
    span: Span;
  }
  /** Any other IR statement kind — rejected by the planner. */
  export interface OtherStmt {
    kind:
      | 'ExprStmt' | 'If' | 'While' | 'ReturnFromFunction' | 'Break'
      | 'Continue' | 'TypeComment' | 'MemberStore'
      | 'IndexStore' | 'IndexSliceStore' | 'CellIndexStore';
    span: Span;
  }

  export type IRStmt = Assign | For | MultiAssignCall | OtherStmt;

  export interface IRFunc {
    name: string;
    cName: string;
    /** Parameter source names. */
    params: string[];
    /** Parameter cNames, parallel to `params`. */
    cParams: string[];
    paramTypes: Type[];
    /** Output source names. */
    outputs: string[];
    /** Output cNames, parallel to `outputs`. */
    cOutputs: string[];
    outputTypes: Type[];
    body: IRStmt[];
    span: Span;
  }

  export interface IRProgram {
    topLevelStmts: IRStmt[];
    functions: Map<string, IRFunc>;
  }
}

declare module 'numbl-src/numbl-core/parser/index.ts' {
  export interface AbstractSyntaxTree {
    body: unknown[];
  }
  export function parseMFile(input: string, fileName?: string): AbstractSyntaxTree;
  export class SyntaxError extends Error {}
}

declare module 'numbl-src/numbl-core/jit/index.ts' {
  import type { AbstractSyntaxTree } from 'numbl-src/numbl-core/parser/index.ts';
  import type { IRProgram, IRFunc, Span } from 'numbl-src/numbl-core/jit/lowering/ir.ts';
  import type { Type, NumericType, Sign } from 'numbl-src/numbl-core/jit/lowering/types.ts';

  export interface WorkspaceFile {
    name: string;
    source: string;
    ast?: AbstractSyntaxTree;
  }

  export class Workspace {
    constructor(mainFile: string, searchPaths?: ReadonlyArray<string>);
    addFile(file: WorkspaceFile): void;
    finalize(): void;
  }

  export interface EnvEntry {
    cName: string;
    ty: Type;
    maybeUnassigned?: boolean;
  }

  export class Lowerer {
    constructor(workspace: Workspace);
    /** Pre-bindable variable scope: seed host-provided values here. */
    env: Map<string, EnvEntry>;
    specializations: Map<string, IRFunc>;
    lowerProgram(ast: AbstractSyntaxTree): IRProgram;
  }

  /** Thrown for MATLAB the JIT pipeline cannot lower; carries a source span. */
  export class UnsupportedConstruct extends Error {
    span?: Span;
  }
  export class JitTypeError extends Error {
    span?: Span;
  }

  export function tensorDouble(shape: number[], exact?: Float64Array): NumericType;
  export function scalarDouble(sign?: Sign, exact?: number): NumericType;
  export function isMultiElement(t: NumericType): boolean;
}

declare module 'numbl-src/numbl-core/jit/lowering/specialize.ts' {
  import type { Lowerer } from 'numbl-src/numbl-core/jit/index.ts';
  import type { IRFunc, IRExpr, Span } from 'numbl-src/numbl-core/jit/lowering/ir.ts';
  import type { Type } from 'numbl-src/numbl-core/jit/lowering/types.ts';

  /**
   * Lower one user function for a concrete argument-type signature. Called with
   * a `Lowerer` as `this` (numbl's own JIT does the same), so specializations
   * accumulate in `lowerer.specializations`.
   */
  export function specializeUserFunction(
    this: Lowerer,
    decl: unknown,
    argTypes: Type[],
    specSource?: string,
    definingFile?: string,
    preSeedOutput?: { name: string; ty: Type; initExpr: IRExpr },
    nargout?: number,
    callSiteSpan?: Span,
  ): IRFunc;
}

declare module 'numbl-src/numbl-core/jit/codegen/inlinePass.ts' {
  import type { IRProgram } from 'numbl-src/numbl-core/jit/lowering/ir.ts';
  /** Folds single-use ANF temps into their consumer, in place. */
  export function inlinePass(prog: IRProgram): void;
}

declare module 'numbl-src/numbl-core/jit/builtins/index.ts' {
  export interface Builtin {
    name: string;
    /** Safe to evaluate one output element from one input element per slot. */
    elementwise?: boolean;
  }
  export function getBuiltin(name: string): Builtin | undefined;
}
