use super::*;

pub struct SGD;

impl Optimizer for SGD {
    fn update(
        &mut self,
        params: &mut HashMap<String, Array2<f32>>,
        grads: &HashMap<String, Array2<f32>>,
        learning_rate: f32,
    ) {
        for (key, param) in params.iter_mut() {
            let grad = grads.get(key).unwrap();
            *param -= &(learning_rate * grad);
        }
    }
}
