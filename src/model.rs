//! Single-layer predictive-coding sequence model.
//!
//! A latent vector `z` generates a window of `context + 1` one-hot symbols:
//! the last `context` symbols and the one that follows them.
//!
//!   window_hat = W · tanh(z)
//!
//! *Predicting*: clamp the observed context part of the window, relax `z` to
//! minimise the context prediction error, then read the next-symbol slot.
//! *Learning*: once the true next symbol arrives, relax `z` against the full
//! window and nudge `W` by (error × activity). Every update is local.

use crate::mat::Mat;
use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab: usize,
    pub context: usize,
    pub hidden: usize,
    pub infer_steps: usize,
    pub infer_rate: f32,
    pub learn_rate: f32,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            vocab: 12,
            context: 3,
            hidden: 24,
            infer_steps: 30,
            infer_rate: 0.2,
            learn_rate: 0.05,
            seed: 1,
        }
    }
}

/// Result of feeding one symbol to the model.
#[derive(Clone, Debug)]
pub struct Step {
    /// Probability distribution the model had for this symbol (None until
    /// `context` symbols have been seen).
    pub predicted: Option<Vec<f32>>,
    /// Surprise in bits: -log2 p(symbol). 0.0 while warming up.
    pub surprise: f32,
}

pub struct PcModel {
    cfg: Config,
    w: Mat,
    history: Vec<usize>,
}

fn softmax(v: &[f32], temp: f32) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|x| ((x - max) / temp).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

impl PcModel {
    pub fn new(cfg: Config) -> Self {
        assert!(cfg.vocab > 0 && cfg.context > 0 && cfg.hidden > 0);
        let dim = (cfg.context + 1) * cfg.vocab;
        let mut rng = Rng::new(cfg.seed);
        let mut w = Mat::zeros(dim, cfg.hidden);
        let scale = 1.0 / (cfg.hidden as f32).sqrt();
        w.data.iter_mut().for_each(|x| *x = rng.symmetric(scale));
        PcModel {
            cfg,
            w,
            history: Vec::new(),
        }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    fn window_dim(&self) -> usize {
        (self.cfg.context + 1) * self.cfg.vocab
    }

    fn encode(&self, symbols: &[usize]) -> Vec<f32> {
        let mut x = vec![0.0; self.window_dim()];
        for (slot, &s) in symbols.iter().enumerate() {
            assert!(s < self.cfg.vocab, "symbol {s} out of vocabulary");
            x[slot * self.cfg.vocab + s] = 1.0;
        }
        x
    }

    /// Relax the latent state against the first `observed` dims of `x`.
    /// Returns (z, activity = tanh(z)).
    fn infer(&self, x: &[f32], observed: usize) -> (Vec<f32>, Vec<f32>) {
        let h = self.cfg.hidden;
        let mut z: Vec<f32> = vec![0.0; h];
        let mut act = vec![0.0; h];
        let mut pred = vec![0.0; self.window_dim()];
        let mut back = vec![0.0; h];
        for _ in 0..self.cfg.infer_steps {
            act.iter_mut().zip(&z).for_each(|(a, zv)| *a = zv.tanh());
            self.w.mul_vec(&act, &mut pred);
            for (i, p) in pred.iter_mut().enumerate() {
                *p = if i < observed { x[i] - *p } else { 0.0 };
            }
            self.w.tmul_vec(&pred, &mut back);
            for j in 0..h {
                let d = 1.0 - act[j] * act[j];
                z[j] += self.cfg.infer_rate * (back[j] * d - 0.01 * z[j]);
            }
        }
        act.iter_mut().zip(&z).for_each(|(a, zv)| *a = zv.tanh());
        (z, act)
    }

    /// Probability of each next symbol given exactly `context` symbols.
    pub fn predict(&self, context: &[usize]) -> Vec<f32> {
        assert_eq!(context.len(), self.cfg.context);
        let x = self.encode(context);
        let observed = self.cfg.context * self.cfg.vocab;
        let (_, act) = self.infer(&x, observed);
        let mut pred = vec![0.0; self.window_dim()];
        self.w.mul_vec(&act, &mut pred);
        softmax(&pred[observed..], 0.25)
    }

    /// Learn from one full window (context followed by its next symbol).
    fn learn(&mut self, window: &[usize]) {
        let x = self.encode(window);
        let dim = self.window_dim();
        let (_, act) = self.infer(&x, dim);
        let mut pred = vec![0.0; dim];
        self.w.mul_vec(&act, &mut pred);
        let err: Vec<f32> = x.iter().zip(&pred).map(|(a, b)| a - b).collect();
        self.w.add_outer(self.cfg.learn_rate, &err, &act);
    }

    /// Feed the next symbol of the stream: predict it, score the surprise,
    /// then learn from it.
    pub fn observe(&mut self, symbol: usize) -> Step {
        assert!(symbol < self.cfg.vocab, "symbol {symbol} out of vocabulary");
        let k = self.cfg.context;
        let mut step = Step {
            predicted: None,
            surprise: 0.0,
        };
        if self.history.len() >= k {
            let ctx: Vec<usize> = self.history[self.history.len() - k..].to_vec();
            let probs = self.predict(&ctx);
            step.surprise = -(probs[symbol].max(1e-9)).log2();
            step.predicted = Some(probs);
            let mut window = ctx;
            window.push(symbol);
            self.learn(&window);
        }
        self.history.push(symbol);
        if self.history.len() > k {
            self.history.remove(0);
        }
        step
    }

    /// Forget the recent stream (keeps learned weights).
    pub fn reset_context(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    fn melody() -> Vec<usize> {
        // Twinkle-ish phrase; every 3-symbol context is unique in the loop.
        vec![0, 0, 4, 4, 5, 5, 4, 3, 3, 2, 2, 1, 1, 0]
    }

    #[test]
    fn probabilities_sum_to_one() {
        let m = PcModel::new(Config::default());
        let p = m.predict(&[0, 1, 2]);
        assert_eq!(p.len(), 12);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn warmup_steps_report_no_prediction() {
        let mut m = PcModel::new(Config::default());
        for s in [0, 1, 2] {
            let st = m.observe(s);
            assert!(st.predicted.is_none());
            assert_eq!(st.surprise, 0.0);
        }
        assert!(m.observe(3).predicted.is_some());
    }

    #[test]
    fn learns_a_repeating_melody_online() {
        let mut m = PcModel::new(Config::default());
        let tune = melody();
        let mut first_pass = 0.0;
        let mut last_pass = 0.0;
        for rep in 0..120 {
            let mut total = 0.0;
            for &s in &tune {
                total += m.observe(s).surprise;
            }
            if rep == 1 {
                first_pass = total;
            }
            if rep == 119 {
                last_pass = total;
            }
        }
        assert!(
            last_pass < first_pass * 0.25,
            "surprise should collapse: first {first_pass}, last {last_pass}"
        );
    }

    #[test]
    fn after_training_predictions_match_the_melody() {
        let mut m = PcModel::new(Config::default());
        let tune = melody();
        for _ in 0..200 {
            for &s in &tune {
                m.observe(s);
            }
        }
        let n = tune.len();
        let mut correct = 0;
        for i in 0..n {
            let ctx = [tune[i], tune[(i + 1) % n], tune[(i + 2) % n]];
            if argmax(&m.predict(&ctx)) == tune[(i + 3) % n] {
                correct += 1;
            }
        }
        assert!(correct >= n - 1, "only {correct}/{n} next notes predicted");
    }

    #[test]
    fn same_seed_gives_identical_learning() {
        let run = || {
            let mut m = PcModel::new(Config::default());
            let mut s = 0.0;
            for _ in 0..10 {
                for x in melody() {
                    s += m.observe(x).surprise;
                }
            }
            s
        };
        assert_eq!(run(), run());
    }

    #[test]
    #[should_panic]
    fn rejects_out_of_vocabulary_symbols() {
        PcModel::new(Config::default()).observe(99);
    }
}
