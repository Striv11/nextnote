//! Day-1 terminal demo: watch surprise collapse as the model learns a tune.

use nextnote::{Config, PcModel};

fn main() {
    let tune = [0usize, 0, 4, 4, 5, 5, 4, 3, 3, 2, 2, 1, 1, 0];
    let mut model = PcModel::new(Config::default());
    println!("pass  surprise(bits)  bar");
    for pass in 1..=60 {
        let total: f32 = tune.iter().map(|&s| model.observe(s).surprise).sum();
        if pass <= 5 || pass % 10 == 0 {
            let bar = "#".repeat((total / 2.0).round().clamp(0.0, 60.0) as usize);
            println!("{pass:>4}  {total:>14.2}  {bar}");
        }
    }
}
