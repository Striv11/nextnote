// Shared golden value: FNV-1a over the bit patterns of every per-note
// surprise (f32) for 30 passes of the demo tune. Must equal the Rust test.
export const TUNE = [0, 0, 4, 4, 5, 5, 4, 3, 3, 2, 2, 1, 1, 0];
export const PASSES = 30;
export function fnv(bits) {
  let h = 0x811c9dc5;
  for (const b of bits) for (let i = 0; i < 4; i++) {
    h ^= (b >>> (8 * i)) & 0xff;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}
