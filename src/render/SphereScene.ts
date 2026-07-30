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
 * Adapted from figpack's SphereEmbedding view (figpack_experimental).
 */
export class SphereScene {
  #scene: THREE.Scene;
  #camera: THREE.PerspectiveCamera;
  #renderer: THREE.WebGLRenderer;
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

    this.#renderer = new THREE.WebGLRenderer({ antialias: true });
    this.#renderer.setPixelRatio(window.devicePixelRatio || 1);
    // The canvas always fills its container via CSS; resize() then only
    // updates the drawing buffer
    this.#renderer.domElement.style.width = '100%';
    this.#renderer.domElement.style.height = '100%';
    this.#renderer.domElement.style.display = 'block';
    container.appendChild(this.#renderer.domElement);

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

    this.#controls = new OrbitControls(this.#camera, this.#renderer.domElement);
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
    this.#renderer.render(this.#scene, this.#camera);
  };

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

  /** The renderer's canvas, for capturing frames. */
  get canvas(): HTMLCanvasElement {
    return this.#renderer.domElement;
  }

  /**
   * Render immediately, outside the animation loop. A WebGL canvas without
   * preserveDrawingBuffer keeps its drawing buffer only until the browser next
   * composites, so a capturer must render and copy within one task.
   */
  renderNow(): void {
    this.#needsRender = false;
    this.#renderer.render(this.#scene, this.#camera);
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
    // updateStyle=false: the canvas keeps its 100%/100% CSS sizing
    this.#renderer.setSize(width, height, false);
    // setSize clears the drawing buffer, so a re-render is required even
    // though nothing in the scene moved
    this.#needsRender = true;
  }

  /**
   * Set the drawing buffer to an exact square pixel size, independent of the
   * container and devicePixelRatio — for capturing at a chosen resolution.
   * The canvas keeps its CSS sizing, so on screen it just rescales. Undo with
   * restoreSize().
   */
  captureSize(px: number): void {
    this.#renderer.setPixelRatio(1);
    this.#renderer.setSize(px, px, false);
    this.#camera.aspect = 1;
    this.#camera.updateProjectionMatrix();
    this.#needsRender = true;
  }

  /** Return from captureSize() to the container-driven buffer size. */
  restoreSize(): void {
    this.#renderer.setPixelRatio(window.devicePixelRatio || 1);
    if (this.#lastW > 0 && this.#lastH > 0) {
      this.#renderer.setSize(this.#lastW, this.#lastH, false);
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
    this.#geometry.dispose();
    (this.#mesh.material as THREE.Material).dispose();
    if (this.#renderer.domElement.parentNode) {
      this.#renderer.domElement.parentNode.removeChild(
        this.#renderer.domElement,
      );
    }
    this.#renderer.dispose();
  }
}
