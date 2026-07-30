/**
 * A small MATLAB tokenizer, for syntax highlighting the model editor.
 *
 * Only what highlighting needs — comments, literals, numbers, keywords — and
 * deliberately not a parser: numbl does the real parsing, and reports errors
 * with positions. Tokens preserve the source text exactly, character for
 * character, because the highlighted output is overlaid on a textarea and any
 * dropped or added character would shift the two out of alignment.
 */

export type TokenClass = 'com' | 'str' | 'num' | 'kw' | 'ext';

export interface Token {
  text: string;
  cls: TokenClass | null;
}

const KEYWORDS = new Set([
  'break', 'case', 'catch', 'classdef', 'continue', 'else', 'elseif', 'end',
  'for', 'function', 'global', 'if', 'otherwise', 'parfor', 'persistent',
  'return', 'spmd', 'switch', 'try', 'while',
]);

const isIdentStart = (c: string): boolean => /[A-Za-z_]/.test(c);
const isIdent = (c: string): boolean => /[A-Za-z0-9_]/.test(c);
const isDigit = (c: string): boolean => c >= '0' && c <= '9';

/**
 * In MATLAB `'` is both the transpose operator and the char-literal delimiter.
 * It opens a literal unless it directly follows something that can be
 * transposed — a value, a closing bracket, or another transpose.
 */
function quoteIsTranspose(src: string, at: number): boolean {
  for (let i = at - 1; i >= 0; i--) {
    const c = src[i];
    if (c === ' ' || c === '\t') continue;
    return isIdent(c) || c === ')' || c === ']' || c === '}' || c === '.' || c === "'";
  }
  return false;
}

/**
 * Tokenize `src`. `external` names (the operations the host provides, e.g.
 * `synth` / `analys`) get their own class so the boundary between the model and
 * what it is given is visible in the editor.
 */
export function tokenizeMatlab(
  src: string,
  external: ReadonlySet<string> = new Set(),
): Token[] {
  const out: Token[] = [];
  const push = (text: string, cls: TokenClass | null): void => {
    if (!text) return;
    const last = out[out.length - 1];
    if (last && last.cls === cls) last.text += text;
    else out.push({ text, cls });
  };

  let i = 0;
  let atLineStart = true;
  let inBlockComment = false;

  while (i < src.length) {
    const c = src[i];

    // Block comments: `%{` and `%}` each alone on their line.
    if (atLineStart) {
      const eol = src.indexOf('\n', i);
      const lineEnd = eol === -1 ? src.length : eol;
      const line = src.slice(i, lineEnd);
      const trimmed = line.trim();
      if (!inBlockComment && trimmed === '%{') inBlockComment = true;
      else if (inBlockComment && trimmed === '%}') {
        push(line, 'com');
        i = lineEnd;
        inBlockComment = false;
        atLineStart = false;
        continue;
      }
      if (inBlockComment) {
        push(line, 'com');
        i = lineEnd;
        atLineStart = false;
        continue;
      }
    }

    if (c === '\n') {
      push(c, null);
      i++;
      atLineStart = true;
      continue;
    }
    if (c === ' ' || c === '\t') {
      push(c, null);
      i++;
      continue;
    }
    atLineStart = false;

    // Line comment, including MATLAB's `%%` section markers.
    if (c === '%') {
      const eol = src.indexOf('\n', i);
      const end = eol === -1 ? src.length : eol;
      push(src.slice(i, end), 'com');
      i = end;
      continue;
    }

    // Line continuation is an operator, but any trailing text is a comment.
    if (c === '.' && src.startsWith('...', i)) {
      const eol = src.indexOf('\n', i);
      const end = eol === -1 ? src.length : eol;
      push('...', null);
      push(src.slice(i + 3, end), 'com');
      i = end;
      continue;
    }

    // Char literal (or transpose).
    if (c === "'") {
      if (quoteIsTranspose(src, i)) {
        push("'", null);
        i++;
        continue;
      }
      let j = i + 1;
      while (j < src.length && src[j] !== '\n') {
        if (src[j] === "'") {
          if (src[j + 1] === "'") j += 2; // escaped quote
          else {
            j++;
            break;
          }
        } else j++;
      }
      push(src.slice(i, j), 'str');
      i = j;
      continue;
    }

    // Double-quoted string.
    if (c === '"') {
      let j = i + 1;
      while (j < src.length && src[j] !== '\n') {
        if (src[j] === '"') {
          if (src[j + 1] === '"') j += 2;
          else {
            j++;
            break;
          }
        } else j++;
      }
      push(src.slice(i, j), 'str');
      i = j;
      continue;
    }

    // Number: 12, 1.5, .5, 1e-3, 2i
    if (isDigit(c) || (c === '.' && isDigit(src[i + 1]))) {
      let j = i;
      while (j < src.length && isDigit(src[j])) j++;
      if (src[j] === '.') {
        j++;
        while (j < src.length && isDigit(src[j])) j++;
      }
      if (src[j] === 'e' || src[j] === 'E') {
        let k = j + 1;
        if (src[k] === '+' || src[k] === '-') k++;
        if (isDigit(src[k])) {
          k++;
          while (k < src.length && isDigit(src[k])) k++;
          j = k;
        }
      }
      if (src[j] === 'i' || src[j] === 'j') j++;
      push(src.slice(i, j), 'num');
      i = j;
      continue;
    }

    // Identifier / keyword / external operation.
    if (isIdentStart(c)) {
      let j = i;
      while (j < src.length && isIdent(src[j])) j++;
      const word = src.slice(i, j);
      push(word, KEYWORDS.has(word) ? 'kw' : external.has(word) ? 'ext' : null);
      i = j;
      continue;
    }

    push(c, null);
    i++;
  }

  return out;
}

const escapeHtml = (s: string): string =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** Highlighted HTML for `src`, safe to assign to innerHTML. */
export function highlightMatlab(
  src: string,
  external: ReadonlySet<string> = new Set(),
): string {
  const html = tokenizeMatlab(src, external)
    .map((t) => (t.cls ? `<span class="tok-${t.cls}">${escapeHtml(t.text)}</span>` : escapeHtml(t.text)))
    .join('');
  // A trailing newline keeps the last line's box height stable, so the overlay
  // and the textarea scroll to the same extent.
  return `${html}\n`;
}
