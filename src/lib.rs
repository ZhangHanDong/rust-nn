// 优化：引入 rayon 并行计算
use ndarray::prelude::*;
use ndarray::Array2;
use polars::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;
use serde_json;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::path::PathBuf;

mod adam;
mod dnn;
mod sgd;
mod utils;

pub use adam::Adam;
pub use dnn::*;
pub use sgd::SGD;
pub use utils::*;

pub trait Optimizer {
    fn update(
        &mut self,
        params: &mut HashMap<String, Array2<f32>>,
        grads: &HashMap<String, Array2<f32>>,
        learning_rate: f32,
    );
}
