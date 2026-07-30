/**
 * Shared WGSL fragments.  Shaders are generated as strings with all sizes
 * baked in as compile-time constants (the WGSL analog of what SHTNS does
 * with NVRTC on CUDA: cf. init_cuda_program() in sht_gpu.cu).
 *
 * fp32 extended-range constants: same values SHTNS injects for a
 * single-precision recurrence (sht_gpu.cu):
 *   SHT_ACCURACY     = 1e-15
 *   SHT_SCALE_FACTOR = 2^56 = 7.2057594037927936e16
 * A per-thread integer exponent `ny` counts how many times the running
 * Legendre value has been multiplied by SCALE to stay in fp32 range;
 * contributions are only accumulated once ny == 0 (value back in normal
 * range and significant).
 */

export const RESCALE_WGSL = /* wgsl */ `
const SCALE: f32 = 7.2057594e16;        // rounds to exactly 2^56 in f32
const INV_SCALE: f32 = 1.0 / 7.2057594e16;
const ACCURACY: f32 = 1e-15;
const RESCALE_THR: f32 = ACCURACY * SCALE + 1.0;   // ~73: value became significant again

struct Seed { y0: f32, ny: i32 }

// Seed of the recurrence: y0 ~ sin(theta)^m by binary exponentiation with
// rescaling (ports the HI_LLIM path of SHT/cuda_legendre.gen.cu, ~651-691).
// The caller multiplies by amm afterwards (|amm| is O(1)).
fn sinpow_rescaled(st: f32, m: u32) -> Seed {
  var y0: f32 = 1.0;
  var ny: i32 = 0;
  if (m > 0u) {
    var s: f32 = st;
    var lb: u32 = m;
    if ((lb & 1u) != 0u) { y0 = s; }
    var nsint: i32 = 0;
    lb = lb >> 1u;
    while (lb > 0u) {
      s = s * s;
      nsint = nsint + nsint;
      if (s < INV_SCALE) {
        nsint = nsint - 1;
        s = s * SCALE;
      }
      if ((lb & 1u) != 0u) {
        y0 = y0 * s;
        ny = ny + nsint;
        if (y0 < (ACCURACY + INV_SCALE)) {
          y0 = y0 * SCALE;
          ny = ny - 1;
        }
      }
      lb = lb >> 1u;
    }
  }
  return Seed(y0, ny);
}
`;
