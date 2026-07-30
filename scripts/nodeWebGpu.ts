/**
 * Desktop WebGPU for the command-line scripts, via the optional `webgpu`
 * package (prebuilt Google Dawn).
 *
 * Installs Dawn under the globals the transform code expects (navigator.gpu,
 * GPUBufferUsage, ...) so everything under src/ runs here unchanged —
 * including requestShtDevice(), which makes the same device request the
 * browser makes.
 */

export const errMsg = (e: unknown): string =>
  e instanceof Error ? e.message : String(e);

/**
 * Returns a human-readable runtime description. The import specifier is
 * indirect so typechecking does not require the optional package.
 */
export async function installWebGpu(): Promise<string> {
  const specifier = 'webgpu';
  let mod: {
    create: (flags: string[]) => GPU;
    globals: Record<string, unknown>;
  };
  try {
    mod = await import(specifier);
  } catch (e) {
    // Distinguish "not installed" from "installed but the prebuilt Dawn binary
    // will not load" — the second is what a machine missing a system library
    // looks like, and reporting it as the first sends people in circles.
    const detail = errMsg(e);
    if (/Cannot find (package|module) '?webgpu'?/.test(detail)) {
      throw new Error(
        'desktop WebGPU needs the optional `webgpu` package (prebuilt Google Dawn):\n' +
          '  npm install webgpu\n' +
          'It is an optionalDependency, so npm can skip it silently — `npm ls webgpu`\n' +
          'says whether it is there.',
      );
    }
    const glibc = /GLIBC_([0-9.]+)/.exec(detail);
    throw new Error(
      `the \`webgpu\` package is installed but did not load:\n  ${detail}\n` +
        (glibc
          ? `Dawn's prebuilt binary wants glibc ${glibc[1]} or newer and this host is older\n` +
            '(`ldd --version` says how old). No flag bridges that — use a container with a\n' +
            'newer base image, or a newer host.\n'
          : 'That is usually the prebuilt Dawn binary missing a system library.\n'),
    );
  }
  Object.assign(globalThis, mod.globals);
  // DAWN_FLAGS is ';'-separated because individual Dawn options take
  // comma-separated lists, e.g. 'enable-dawn-features=allow_unsafe_apis,...'
  const dawnFlags = process.env.DAWN_FLAGS?.split(';').filter(Boolean) ?? [];
  Object.defineProperty(globalThis, 'navigator', {
    value: { gpu: mod.create(dawnFlags) },
    configurable: true,
    writable: true,
  });
  const { version } = await import(`${specifier}/package.json`, {
    with: { type: 'json' },
  }).then(
    (m) => m.default as { version: string },
    () => ({ version: '?' }),
  );
  return `node-webgpu ${version} (Google Dawn)`;
}

/** The hint to print when Dawn loads but finds no adapter. */
export const NO_ADAPTER_HINT =
  '  Dawn reaches the GPU through Vulkan on Linux and Windows, Metal on macOS,\n' +
  "  so a headless box may have no adapter at all. DAWN_FLAGS='backend=vulkan'\n" +
  '  makes it explain itself.';
