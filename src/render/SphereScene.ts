import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * Three.js scene wrapper: a single indexed triangle mesh with dynamic
 * per-vertex positions and colors, orbit controls, and optional camera
 * synchronization with sibling scenes. The topology is fixed by the grid; the
 * positions are the surface, so they change when the geometry or the morph
 * does, and the colors every frame.
 *
 * Rendering is on demand: the animation loop ticks every frame (it has to,
 * to drive OrbitControls damping), but only re-renders when the colors,
 * camera, or canvas size actually changed.
 *
 * Every scene draws with one page-wide WebGL renderer (see sharedRenderer)
 * and copies the result into its own 2D canvas, which is what sits in the
 * page. A renderer per scene would be a WebGL context per panel, and browsers
 * keep only about 16 of those alive per page (Chrome drops the oldest past
 * that), so a compare grid of more panels would go blank from the top. The
 * copy is a GPU-side blit of the panel's pixels, paid only on the frames that
 * render at all; and because the 2D canvas keeps its contents, a capturer can
 * read it at any time, not only in the task that rendered.
 *
 * Adapted from figpack's SphereEmbedding view (figpack_experimental).
 */

let shared: THREE.WebGLRenderer | null = null;

/**
 * The one WebGL renderer every scene draws with, its canvas grown (never
 * shrunk) to at least `w` x `h` device pixels. Scenes render into its
 * bottom-left `w` x `h` corner under a scissor, so a smaller panel after a
 * larger one costs no reallocation. It lives for the page: disposing it with
 * the last scene would only recompile the same shaders on the next rebuild.
 * Pixel ratio is 1 because each scene hands it device pixels already.
 */
function sharedRenderer(w: number, h: number): THREE.WebGLRenderer {
  if (!shared) {
    shared = new THREE.WebGLRenderer({ antialias: true });
    shared.setPixelRatio(1);
    shared.setScissorTest(true);
  }
  const c = shared.domElement;
  if (c.width < w || c.height < h) {
    // updateStyle=false: the canvas is never in the page.
    shared.setSize(Math.max(c.width, w), Math.max(c.height, h), false);
  }
  return shared;
}

export class SphereScene {
  #scene: THREE.Scene;
  #camera: THREE.PerspectiveCamera;
  /** What the page shows: this scene's pixels, copied from the shared renderer. */
  #canvas: HTMLCanvasElement;
  #ctx: CanvasRenderingContext2D;
  #controls: OrbitControls;
  #geometry: THREE.BufferGeometry;
  #mesh: THREE.Mesh;
  #animationId: number | null = null;
  #defaultCameraState: {
    position: THREE.Vector3;
    target: THREE.Vector3;
  } | null = null;
  #syncing = false;
  #needsRender = true;
  #lastW = -1;
  #lastH = -1;

  constructor(
    container: HTMLElement,
    numVertices: number,
    indices: Uint32Array,
    positions: Float32Array,
    background = '#14161c',
  ) {
    this.#scene = new THREE.Scene();
    this.#scene.background = new THREE.Color(background);

    this.#camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);

    this.#canvas = document.createElement('canvas');
    const ctx = this.#canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('no 2d context for a sphere panel');
    this.#ctx = ctx;
    // The canvas always fills its container via CSS; resize() then only
    // updates the drawing buffer
    this.#canvas.style.width = '100%';
    this.#canvas.style.height = '100%';
    this.#canvas.style.display = 'block';
    container.appendChild(this.#canvas);

    // Lighting: ambient plus a headlight attached to the camera so the
    // surface stays lit from the viewing direction as it is rotated
    this.#scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const headlight = new THREE.DirectionalLight(0xffffff, 1.6);
    headlight.position.set(0.5, 0.8, 1);
    this.#camera.add(headlight);
    this.#scene.add(this.#camera);

    this.#geometry = new THREE.BufferGeometry();
    // Positions move with the morph slider, so they are dynamic too.
    const positionAttr = new THREE.BufferAttribute(positions, 3);
    positionAttr.setUsage(THREE.DynamicDrawUsage);
    const colorAttr = new THREE.BufferAttribute(
      new Float32Array(numVertices * 3),
      3,
    );
    colorAttr.setUsage(THREE.DynamicDrawUsage);
    this.#geometry.setAttribute('position', positionAttr);
    this.#geometry.setAttribute('color', colorAttr);
    this.#geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    this.#geometry.computeVertexNormals();
    this.#geometry.computeBoundingSphere();

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: 25,
      specular: new THREE.Color(0x222222),
    });
    this.#mesh = new THREE.Mesh(this.#geometry, material);
    this.#scene.add(this.#mesh);

    this.#controls = new OrbitControls(this.#camera, this.#canvas);
    this.#controls.enableDamping = true;
    this.#controls.dampingFactor = 0.1;
    // Fires on user input and on every damping-tail update, so the flag stays
    // set until the camera has fully settled.
    this.#controls.addEventListener('change', () => {
      this.#needsRender = true;
    });

    this.#animate();
  }

  #animate = () => {
    this.#animationId = requestAnimationFrame(this.#animate);
    this.#controls.update();
    if (!this.#needsRender) return;
    this.#needsRender = false;
    this.#render();
  };

  /**
   * Draw into the shared renderer's bottom-left corner and copy that corner
   * out. The copy has to happen in the same task as the render: the shared
   * canvas keeps no drawing buffer between tasks, and the next scene reuses
   * the corner anyway. WebGL's origin is bottom-left and the 2D canvas's is
   * top-left, hence the source y.
   */
  #render(): void {
    const w = this.#canvas.width;
    const h = this.#canvas.height;
    if (w === 0 || h === 0) return;
    const r = sharedRenderer(w, h);
    r.setViewport(0, 0, w, h);
    r.setScissor(0, 0, w, h);
    r.render(this.#scene, this.#camera);
    this.#ctx.drawImage(r.domElement, 0, r.domElement.height - h, w, h, 0, 0, w, h);
  }

  updateColors(colors: Float32Array): void {
    const attr = this.#geometry.getAttribute('color') as THREE.BufferAttribute;
    (attr.array as Float32Array).set(colors);
    attr.needsUpdate = true;
    this.#needsRender = true;
  }

  /**
   * Move the vertices — for the sphere/surface morph. Normals have to be
   * recomputed with them or the shading stays that of the old shape, which is
   * the whole thing the eye reads a curved surface by.
   */
  updatePositions(positions: Float32Array): void {
    const attr = this.#geometry.getAttribute('position') as THREE.BufferAttribute;
    (attr.array as Float32Array).set(positions);
    attr.needsUpdate = true;
    this.#geometry.computeVertexNormals();
    this.#geometry.computeBoundingSphere();
    this.#needsRender = true;
  }

  /** The panel's canvas (2D, holding the last render), for capturing frames. */
  get canvas(): HTMLCanvasElement {
    return this.#canvas;
  }

  /** Render immediately, outside the animation loop — for a capturer that
   *  has just changed the scene and wants the canvas current now. */
  renderNow(): void {
    this.#needsRender = false;
    this.#render();
  }

  /** Mirror this scene's camera whenever the other scene's controls move. */
  syncCamerasWith(other: SphereScene): void {
    const follow = (src: SphereScene, dst: SphereScene) => {
      src.#controls.addEventListener('change', () => {
        if (dst.#syncing) return;
        src.#syncing = true;
        dst.#camera.position.copy(src.#camera.position);
        dst.#camera.zoom = src.#camera.zoom;
        dst.#camera.updateProjectionMatrix();
        dst.#controls.target.copy(src.#controls.target);
        dst.#controls.update();
        dst.#needsRender = true;
        src.#syncing = false;
      });
    };
    follow(this, other);
    follow(other, this);
  }

  /** Orbit the camera about the up axis by `angle` radians, keeping the
   *  target. Synced sibling scenes follow via their controls, as with a drag. */
  orbitBy(angle: number): void {
    const offset = this.#camera.position.clone().sub(this.#controls.target);
    offset.applyAxisAngle(this.#camera.up, angle);
    this.#camera.position.copy(this.#controls.target).add(offset);
    this.#controls.update();
    this.#needsRender = true;
  }

  /** Camera pose, for carrying the view across a scene rebuild. */
  cameraState(): { position: THREE.Vector3; target: THREE.Vector3; zoom: number } {
    return {
      position: this.#camera.position.clone(),
      target: this.#controls.target.clone(),
      zoom: this.#camera.zoom,
    };
  }

  setCameraState(s: {
    position: THREE.Vector3;
    target: THREE.Vector3;
    zoom: number;
  }): void {
    this.#camera.position.copy(s.position);
    this.#camera.zoom = s.zoom;
    this.#camera.updateProjectionMatrix();
    this.#controls.target.copy(s.target);
    this.#controls.update();
    this.#needsRender = true;
  }

  /**
   * Position the camera to comfortably frame the geometry.
   *
   * The distance is generous on purpose. The bounding sphere is of the surface
   * currently loaded, but the camera is *kept* across a geometry change and
   * across the morph, so a frame that only just fits the shape at hand would
   * clip the next one. Leaving room means switching shapes never needs a
   * camera reset to see what happened.
   */
  fitCamera(): void {
    this.#geometry.computeBoundingSphere();
    const bs = this.#geometry.boundingSphere;
    if (!bs) return;
    const radius = Math.max(bs.radius, 1e-6);
    const distance = radius * 3.4;
    this.#controls.target.copy(bs.center);
    this.#camera.position.set(
      bs.center.x + distance * 0.55,
      bs.center.y + distance * 0.35,
      bs.center.z + distance * 0.75,
    );
    this.#camera.near = radius * 0.01;
    this.#camera.far = radius * 100;
    this.#camera.updateProjectionMatrix();
    this.#controls.update();
    this.#needsRender = true;
    this.#defaultCameraState = {
      position: this.#camera.position.clone(),
      target: this.#controls.target.clone(),
    };
  }

  resetCamera(): void {
    if (this.#defaultCameraState) {
      this.#camera.position.copy(this.#defaultCameraState.position);
      this.#controls.target.copy(this.#defaultCameraState.target);
      this.#controls.update();
      this.#needsRender = true;
    } else {
      this.fitCamera();
    }
  }

  resize(width: number, height: number): void {
    // Setting canvas.width clears the canvas even at the same value, which
    // shows as a blank flash until the next render — skip no-op resizes.
    if (width === this.#lastW && height === this.#lastH) return;
    this.#lastW = width;
    this.#lastH = height;
    this.#camera.aspect = width / Math.max(1, height);
    this.#camera.updateProjectionMatrix();
    // The canvas keeps its 100%/100% CSS sizing; only the buffer changes.
    this.#setBufferSize(width, height, window.devicePixelRatio || 1);
    // Resizing clears the canvas, so a re-render is required even though
    // nothing in the scene moved
    this.#needsRender = true;
  }

  /** The drawing buffer at `width` x `height` CSS pixels times `ratio`,
   *  floored as three's own setSize does. */
  #setBufferSize(width: number, height: number, ratio: number): void {
    this.#canvas.width = Math.floor(width * ratio);
    this.#canvas.height = Math.floor(height * ratio);
  }

  /**
   * Set the drawing buffer to an exact square pixel size, independent of the
   * container and devicePixelRatio — for capturing at a chosen resolution.
   * The canvas keeps its CSS sizing, so on screen it just rescales. Undo with
   * restoreSize().
   */
  captureSize(px: number): void {
    this.#setBufferSize(px, px, 1);
    this.#camera.aspect = 1;
    this.#camera.updateProjectionMatrix();
    this.#needsRender = true;
  }

  /** Return from captureSize() to the container-driven buffer size. */
  restoreSize(): void {
    if (this.#lastW > 0 && this.#lastH > 0) {
      this.#setBufferSize(this.#lastW, this.#lastH, window.devicePixelRatio || 1);
      this.#camera.aspect = this.#lastW / Math.max(1, this.#lastH);
      this.#camera.updateProjectionMatrix();
    }
    this.#needsRender = true;
  }

  dispose(): void {
    if (this.#animationId !== null) {
      cancelAnimationFrame(this.#animationId);
      this.#animationId = null;
    }
    this.#controls.dispose();
    // Frees their buffers and program use in the shared renderer, which
    // itself stays up for the next scene.
    this.#geometry.dispose();
    (this.#mesh.material as THREE.Material).dispose();
    this.#canvas.remove();
  }
}
