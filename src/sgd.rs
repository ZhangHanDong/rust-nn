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
            if let Some(grad) = grads.get(&format!("d{}", key)) {
                *param -= &(learning_rate * grad);
            } else {
                eprintln!("Gradient for parameter '{}' not found", key);
            }
        }
    }
}
