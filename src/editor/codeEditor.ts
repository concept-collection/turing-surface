/**
 * A textarea with syntax highlighting, by overlay.
 *
 * A textarea cannot colour its own text, so the highlighted source is rendered
 * into a <pre> underneath and the textarea sits on top with transparent text and
 * a visible caret. The two must agree on every metric that affects layout —
 * font, line height, padding, tab size, wrapping — and their scroll offsets are
 * kept in sync, or the colours drift away from the characters.
 */
import { highlightMatlab } from './matlab.ts';

export interface CodeEditorOptions {
  textarea: HTMLTextAreaElement;
  /** The <pre> behind it, holding the highlighted copy. */
  overlay: HTMLElement;
  /** Names to mark as host-provided operations. */
  external?: ReadonlySet<string>;
  /** Called on every edit. */
  onInput?: (value: string) => void;
}

export class CodeEditor {
  #textarea: HTMLTextAreaElement;
  #overlay: HTMLElement;
  #external: ReadonlySet<string>;

  constructor(opts: CodeEditorOptions) {
    this.#textarea = opts.textarea;
    this.#overlay = opts.overlay;
    this.#external = opts.external ?? new Set();

    this.#textarea.addEventListener('input', () => {
      this.#repaint();
      opts.onInput?.(this.#textarea.value);
    });
    // Keep the colours under the characters while scrolling.
    this.#textarea.addEventListener('scroll', () => this.#syncScroll());
    // Tab should indent rather than leave the editor.
    this.#textarea.addEventListener('keydown', (e) => this.#onKeyDown(e));
    this.#repaint();
  }

  get value(): string {
    return this.#textarea.value;
  }

  set value(next: string) {
    this.#textarea.value = next;
    this.#repaint();
  }

  focus(): void {
    this.#textarea.focus();
  }

  /** Select a character range, scrolling it into view. */
  select(start: number, end: number): void {
    this.#textarea.focus();
    this.#textarea.setSelectionRange(start, end);
    // setSelectionRange does not always scroll; nudge the line into view.
    const line = this.#textarea.value.slice(0, start).split('\n').length - 1;
    const lineHeight = this.#textarea.scrollHeight / Math.max(1, this.#lineCount());
    const target = line * lineHeight - this.#textarea.clientHeight / 2;
    this.#textarea.scrollTop = Math.max(0, target);
    this.#syncScroll();
  }

  #lineCount(): number {
    return this.#textarea.value.split('\n').length + 1; // +1 for the trailing line
  }

  #onKeyDown(e: KeyboardEvent): void {
    if (e.key !== 'Tab' || e.ctrlKey || e.metaKey || e.altKey) return;
    e.preventDefault();
    const el = this.#textarea;
    const { selectionStart: s, selectionEnd: t, value } = el;
    el.value = `${value.slice(0, s)}  ${value.slice(t)}`;
    el.selectionStart = el.selectionEnd = s + 2;
    // Let the input listener repaint and notify, as for any other edit.
    el.dispatchEvent(new Event('input'));
  }

  #repaint(): void {
    this.#overlay.innerHTML = highlightMatlab(this.#textarea.value, this.#external);
    this.#syncScroll();
  }

  #syncScroll(): void {
    this.#overlay.scrollTop = this.#textarea.scrollTop;
    this.#overlay.scrollLeft = this.#textarea.scrollLeft;
  }
}
