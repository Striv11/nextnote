//! Minimal dense row-major matrix, just the operations predictive coding needs.

#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    /// out = self * x   (x has `cols` entries, out has `rows`)
    pub fn mul_vec(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        for (r, o) in out.iter_mut().enumerate() {
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
        }
    }

    /// out = selfᵀ * y   (y has `rows` entries, out has `cols`)
    pub fn tmul_vec(&self, y: &[f32], out: &mut [f32]) {
        assert_eq!(y.len(), self.rows);
        assert_eq!(out.len(), self.cols);
        out.iter_mut().for_each(|o| *o = 0.0);
        for (r, &yr) in y.iter().enumerate() {
            if yr == 0.0 {
                continue;
            }
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            for (o, w) in out.iter_mut().zip(row) {
                *o += w * yr;
            }
        }
    }

    /// self += scale * y xᵀ  (a local, Hebbian-style outer-product update)
    pub fn add_outer(&mut self, scale: f32, y: &[f32], x: &[f32]) {
        assert_eq!(y.len(), self.rows);
        assert_eq!(x.len(), self.cols);
        for (r, &yr) in y.iter().enumerate() {
            let s = scale * yr;
            if s == 0.0 {
                continue;
            }
            let row = &mut self.data[r * self.cols..(r + 1) * self.cols];
            for (w, xv) in row.iter_mut().zip(x) {
                *w += s * xv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Mat {
        Mat {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    }

    #[test]
    fn mul_vec_matches_hand_computation() {
        let mut out = [0.0; 2];
        sample().mul_vec(&[1.0, 0.0, -1.0], &mut out);
        assert_eq!(out, [-2.0, -2.0]);
    }

    #[test]
    fn tmul_vec_is_the_transpose_product() {
        let mut out = [0.0; 3];
        sample().tmul_vec(&[1.0, 2.0], &mut out);
        assert_eq!(out, [9.0, 12.0, 15.0]);
    }

    #[test]
    fn add_outer_updates_rank_one() {
        let mut m = Mat::zeros(2, 2);
        m.add_outer(0.5, &[2.0, 0.0], &[1.0, 3.0]);
        assert_eq!(m.data, vec![1.0, 3.0, 0.0, 0.0]);
        assert_eq!(m.get(0, 1), 3.0);
    }
}
