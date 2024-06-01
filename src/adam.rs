use super::*;

pub struct Adam {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: usize,
    pub m: HashMap<String, Array2<f32>>,
    pub v: HashMap<String, Array2<f32>>,
}

impl Adam {
    pub fn new(beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Adam {
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn update(
        &mut self,
        params: &mut HashMap<String, Array2<f32>>,
        grads: &HashMap<String, Array2<f32>>,
        learning_rate: f32,
    ) {
        self.t += 1;
        for (key, param) in params.iter_mut() {
            if let Some(grad) = grads.get(&format!("d{}", key)) {
                let m = self
                    .m
                    .entry(key.clone())
                    .or_insert_with(|| Array2::zeros(param.dim()));
                let v = self
                    .v
                    .entry(key.clone())
                    .or_insert_with(|| Array2::zeros(param.dim()));

                *m = self.beta1 * &*m + (1.0 - self.beta1) * grad;
                *v = self.beta2 * &*v + (1.0 - self.beta2) * grad.mapv(|x| x.powi(2));

                let m_hat = m.mapv(|x| x / (1.0 - self.beta1.powi(self.t as i32)));
                let v_hat = v.mapv(|x| x / (1.0 - self.beta2.powi(self.t as i32)));

                *param -= &(learning_rate * m_hat / (v_hat.mapv(|x| x.sqrt()) + self.epsilon));
            } else {
                eprintln!("Gradient for parameter '{}' not found", key);
            }
        }
    }
}
