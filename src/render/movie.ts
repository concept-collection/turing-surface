/**
 * MP4 recording of the live view.
 *
 * Each frame is composited from the on-screen sphere canvases, so the movie
 * shows what the page shows — current camera orientation, colormap, theme —
 * with a colorbar per species and a caption (model, parameter values, running
 * time). Encoding is WebCodecs H.264 muxed by mp4-muxer, entirely in the
 * browser, so capture runs as fast as the solver recomputes rather than at
 * playback speed.
 */
import { ArrayBufferTarget, Muxer } from 'mp4-muxer';
import type { ColormapFunc } from './colormaps.ts';
import { fmtValue } from './colorbar.ts';

export interface MoviePanel {
  /**
   * The scene's WebGL canvas. It must be rendered in the same task that calls
   * addFrame(): without preserveDrawingBuffer the drawing buffer survives only
   * until the browser next composites.
   */
  canvas: HTMLCanvasElement;
  label: string;
}

/** Per-panel colorbar state for one frame. */
export interface MovieBar {
  cmap: ColormapFunc;
  lo: number;
  hi: number;
}

export interface MovieOptions {
  panels: MoviePanel[];
  /** Caption line 1, bold: the model/preset. */
  title: string;
  /** Caption line 2: the parameter values. */
  subtitle: string;
  /** Playback speed: simulation-time units per second of video. Each frame is
   *  timestamped with its simulation time divided by this, so playback speed
   *  is exact regardless of how many frames the caller captures. */
  speed: number;
  /** Effective frames per second, for encoder rate control only. */
  fps: number;
  /** Rendered edge of each sphere panel, px. The caller renders the scene
   *  canvases at this size; the frame is the panels side by side plus the
   *  caption bar. */
  sphere: number;
}

/**
 * H.264 profile candidates: High, then Main, then Constrained Baseline —
 * Chrome's software fallback encoder supports only the last. The level covers
 * the frame area: 4.0 up to 1080p at 30 fps, 5.1 beyond (large exports).
 */
const h264Candidates = (pixels: number): string[] => {
  const level = pixels <= 1920 * 1080 ? '28' : '33';
  return ['avc1.6400', 'avc1.4d00', 'avc1.42e0'].map((p) => p + level);
};

const even = (x: number): number => 2 * Math.floor(x / 2);

interface Layout {
  /** Sphere panel edge, px. Everything else scales by u = sphere/768. */
  sphere: number;
  u: number;
  /** Colorbar column to the right of each sphere, like the app's. */
  gutter: number;
  captionH: number;
  width: number;
  height: number;
}

/** Sized from the caller's resolution choice, clamped to what H.264 encoders
 *  comfortably handle. Even dimensions, as 4:2:0 encoders require. */
const layoutFor = (nPanels: number, spherePx: number): Layout => {
  const sphere = even(Math.max(240, Math.min(1600, spherePx)));
  const u = sphere / 768;
  const gutter = even(Math.round(72 * u));
  const captionH = even(Math.round(64 * u));
  return {
    sphere,
    u,
    gutter,
    captionH,
    width: nPanels * (sphere + gutter),
    height: sphere + captionH,
  };
};

export class MovieRecorder {
  #panels: MoviePanel[];
  #title: string;
  #subtitle: string;
  #speed: number;
  #fps: number;
  #lastKeyUs = 0;
  #layout: Layout;
  #canvas: HTMLCanvasElement;
  #ctx: CanvasRenderingContext2D;
  #muxer: Muxer<ArrayBufferTarget>;
  #encoder: VideoEncoder;
  #frames = 0;
  #error: unknown = null;
  // Page theme, sampled at creation so the movie matches light/dark mode.
  #bg: string;
  #ink: string;
  #ink2: string;
  #line: string;
  #sphereBg: string;

  static async create(opts: MovieOptions): Promise<MovieRecorder> {
    if (typeof VideoEncoder === 'undefined') {
      throw new Error('WebCodecs is not available in this browser');
    }
    const layout = layoutFor(opts.panels.length, opts.sphere);
    const fps = Math.max(1, Math.round(opts.fps));
    const config = {
      width: layout.width,
      height: layout.height,
      // ~0.15 bits per pixel per frame reads as visually lossless here
      bitrate: Math.min(
        24e6,
        Math.max(2e6, Math.round(layout.width * layout.height * fps * 0.15)),
      ),
      framerate: fps,
    };
    for (const codec of h264Candidates(layout.width * layout.height)) {
      const { supported } = await VideoEncoder.isConfigSupported({ codec, ...config });
      if (supported) return new MovieRecorder(opts, layout, { codec, ...config });
    }
    throw new Error('no supported H.264 encoder configuration');
  }

  private constructor(opts: MovieOptions, layout: Layout, config: VideoEncoderConfig) {
    this.#panels = opts.panels;
    this.#title = opts.title;
    this.#subtitle = opts.subtitle;
    this.#speed = opts.speed;
    this.#fps = Math.max(1, opts.fps);
    this.#layout = layout;

    const css = getComputedStyle(document.documentElement);
    const themeVar = (name: string, fallback: string): string =>
      css.getPropertyValue(name).trim() || fallback;
    this.#bg = themeVar('--bg', '#ffffff');
    this.#ink = themeVar('--ink', '#1f2328');
    this.#ink2 = themeVar('--ink-2', '#57606a');
    this.#line = themeVar('--line', '#d0d7de');
    this.#sphereBg = themeVar('--sphere-bg', '#f4f6f8');

    this.#canvas = document.createElement('canvas');
    this.#canvas.width = layout.width;
    this.#canvas.height = layout.height;
    const ctx = this.#canvas.getContext('2d');
    if (!ctx) throw new Error('no 2d context for the movie canvas');
    this.#ctx = ctx;

    this.#muxer = new Muxer({
      target: new ArrayBufferTarget(),
      video: {
        codec: 'avc',
        width: layout.width,
        height: layout.height,
        frameRate: Math.max(1, Math.round(opts.fps)),
      },
      fastStart: 'in-memory',
    });
    this.#encoder = new VideoEncoder({
      output: (chunk, meta) => this.#muxer.addVideoChunk(chunk, meta),
      error: (e) => (this.#error = e),
    });
    this.#encoder.configure(config);
  }

  /**
   * Composite and encode one frame. The compositing happens synchronously, in
   * the caller's task; the await is only encoder backpressure, so a solver
   * that outruns the encoder does not pile frames up in its queue.
   */
  async addFrame(t: number, bars: MovieBar[]): Promise<void> {
    if (this.#error) throw this.#error;
    this.#compose(t, bars);
    const timestamp = Math.round((t / this.#speed) * 1e6);
    const frame = new VideoFrame(this.#canvas, {
      timestamp,
      duration: Math.round(1e6 / this.#fps),
    });
    // a keyframe every ~2 s of video keeps the file seekable without bloat
    const keyFrame = this.#frames === 0 || timestamp - this.#lastKeyUs >= 2e6;
    if (keyFrame) this.#lastKeyUs = timestamp;
    this.#encoder.encode(frame, { keyFrame });
    frame.close();
    this.#frames++;
    while (this.#encoder.encodeQueueSize > 4) {
      await new Promise((r) => this.#encoder.addEventListener('dequeue', r, { once: true }));
    }
  }

  async finish(): Promise<Blob> {
    await this.#encoder.flush();
    if (this.#error) throw this.#error;
    this.#encoder.close();
    this.#muxer.finalize();
    return new Blob([this.#muxer.target.buffer], { type: 'video/mp4' });
  }

  cancel(): void {
    if (this.#encoder.state !== 'closed') this.#encoder.close();
  }

  // ---------------------------------------------------------------- drawing
  #compose(t: number, bars: MovieBar[]): void {
    const { sphere, gutter } = this.#layout;
    const ctx = this.#ctx;
    ctx.fillStyle = this.#bg;
    ctx.fillRect(0, 0, this.#layout.width, this.#layout.height);
    this.#panels.forEach((panel, k) => {
      const x = k * (sphere + gutter);
      ctx.drawImage(panel.canvas, x, 0, sphere, sphere);
      ctx.fillStyle = this.#sphereBg;
      ctx.fillRect(x + sphere, 0, gutter, sphere);
      this.#drawBar(x + sphere, bars[k]);
      this.#drawTag(x, panel.label);
    });
    this.#drawCaption(t);
  }

  #drawBar(x0: number, bar: MovieBar): void {
    const { sphere, u, gutter } = this.#layout;
    const ctx = this.#ctx;
    const w = Math.round(18 * u);
    const h = Math.round(0.55 * sphere);
    const bx = Math.round(x0 + (gutter - w) / 2);
    const by = Math.round((sphere - h) / 2);
    for (let y = 0; y < h; y++) {
      const [r, g, b] = bar.cmap(1 - y / (h - 1));
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(bx, by + y, w, 1);
    }
    ctx.strokeStyle = this.#line;
    ctx.strokeRect(bx + 0.5, by + 0.5, w - 1, h - 1);
    ctx.fillStyle = this.#ink2;
    ctx.font = `${Math.round(13 * u)}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(fmtValue(bar.hi), x0 + gutter / 2, by - 6 * u);
    ctx.textBaseline = 'top';
    ctx.fillText(fmtValue(bar.lo), x0 + gutter / 2, by + h + 6 * u);
  }

  /** The species name, as the app's floating tag: white on a dark pill. */
  #drawTag(x0: number, label: string): void {
    const { u } = this.#layout;
    const ctx = this.#ctx;
    const size = Math.round(20 * u);
    ctx.font = `600 ${size}px system-ui, sans-serif`;
    const tw = ctx.measureText(label).width;
    const padX = 12 * u;
    const padY = 4 * u;
    const bx = x0 + 12 * u;
    const by = 10 * u;
    const bh = size + 2 * padY;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
    ctx.beginPath();
    ctx.roundRect(bx, by, tw + 2 * padX, bh, bh / 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, bx + padX, by + bh / 2 + u);
  }

  #drawCaption(t: number): void {
    const { sphere, u, captionH, width } = this.#layout;
    const ctx = this.#ctx;
    const pad = 16 * u;
    ctx.strokeStyle = this.#line;
    ctx.beginPath();
    ctx.moveTo(0, sphere + 0.5);
    ctx.lineTo(width, sphere + 0.5);
    ctx.stroke();
    ctx.textBaseline = 'middle';
    ctx.fillStyle = this.#ink;
    ctx.font = `600 ${Math.round(20 * u)}px system-ui, sans-serif`;
    ctx.textAlign = 'left';
    ctx.fillText(this.#title, pad, sphere + captionH * 0.34);
    ctx.font = `${Math.round(20 * u)}px system-ui, sans-serif`;
    ctx.textAlign = 'right';
    ctx.fillText(
      `t = ${t.toFixed(2)} · ${this.#speed}×`,
      width - pad,
      sphere + captionH * 0.34,
    );
    ctx.fillStyle = this.#ink2;
    ctx.font = `${Math.round(14 * u)}px system-ui, sans-serif`;
    ctx.textAlign = 'left';
    ctx.fillText(this.#subtitle, pad, sphere + captionH * 0.74);
  }
}
