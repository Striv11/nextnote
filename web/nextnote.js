// Thin JS wrapper over the nn_* C ABI exported by nextnote.wasm.
export async function load(source) {
  const { instance } = await WebAssembly.instantiate(source, {});
  const x = instance.exports;
  return {
    create(vocab = 12, context = 3, hidden = 24, seed = 1) {
      const h = x.nn_new(vocab, context, hidden, seed);
      if (!h) throw new Error("nn_new: invalid model dimensions");
      const buf = x.nn_alloc(vocab);
      return {
        vocab,
        observe(sym) {
          const s = x.nn_observe(h, sym);
          if (s < 0) throw new Error(`nn_observe: invalid symbol ${sym}`);
          return s;
        },
        /** Float32Array of next-note probabilities, or null while warming up. */
        predict() {
          const n = x.nn_predict(h, buf, vocab);
          if (n < 0) throw new Error("nn_predict failed");
          // Re-create the view each call: wasm memory may have grown.
          return n === 0 ? null : new Float32Array(x.memory.buffer, buf, n).slice();
        },
        resetContext() { x.nn_reset_context(h); },
        free() { x.nn_free(h); x.nn_dealloc(buf, vocab); },
      };
    },
  };
}
