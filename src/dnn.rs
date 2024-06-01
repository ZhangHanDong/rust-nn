use super::*;
use crate::adam::Adam;

pub struct LayerParameters {
    pub weights: Array2<f32>,
    pub biases: Array2<f32>,
}

impl LayerParameters {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let between = Uniform::from(-1.0..1.0);
        let mut rng = rand::thread_rng();

        let weights = Array::from_shape_fn((output_size, input_size), |_| between.sample(&mut rng));
        let biases = Array::from_shape_fn((output_size, 1), |_| 0.0);

        LayerParameters { weights, biases }
    }
}

pub struct DeepNeuralNetwork {
    pub layers: Vec<usize>,
    pub learning_rate: f32,
    pub parameters: HashMap<String, Array2<f32>>,
}

impl DeepNeuralNetwork {
    pub fn new(layers: Vec<usize>, learning_rate: f32) -> Self {
        let parameters = HashMap::new();
        DeepNeuralNetwork {
            layers,
            learning_rate,
            parameters,
        }
    }

    pub fn save_parameters(&self, path: &Path) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &self.parameters)?;
        Ok(())
    }

    pub fn load_parameters(&mut self, path: &Path) -> std::io::Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        self.parameters = serde_json::from_reader(reader)?;
        Ok(())
    }

    pub fn predict(&self, input: &Array2<f32>) -> Array2<f32> {
        let (_, output) = self.forward_propagation(input);
        output
    }

    pub fn initialize_parameters(&mut self) {
        let between = Uniform::from(-1.0..1.0);
        let mut rng = rand::thread_rng();

        for l in 1..self.layers.len() {
            let weight_matrix =
                self.generate_weight_matrix(self.layers[l - 1], self.layers[l], &mut rng, &between);
            let bias_matrix = self.generate_bias_matrix(self.layers[l]);

            let weight_string = format!("W{}", l);
            let biases_string = format!("b{}", l);

            self.parameters.insert(weight_string, weight_matrix);
            self.parameters.insert(biases_string, bias_matrix);
        }
    }

    fn generate_weight_matrix<R: Rng>(
        &self,
        input_size: usize,
        output_size: usize,
        rng: &mut R,
        between: &Uniform<f32>,
    ) -> Array2<f32> {
        let weight_array: Vec<f32> = (0..(input_size * output_size))
            .map(|_| between.sample(rng))
            .collect();
        Array::from_shape_vec((output_size, input_size), weight_array).unwrap()
    }

    fn generate_bias_matrix(&self, output_size: usize) -> Array2<f32> {
        let bias_array: Vec<f32> = vec![0.0; output_size];
        Array::from_shape_vec((output_size, 1), bias_array).unwrap()
    }

    // 优化：隐藏层使用 Relu，输出层使用 Sigmoid
    pub fn forward_propagation(
        &self,
        input: &Array2<f32>,
    ) -> (HashMap<String, Array2<f32>>, Array2<f32>) {
        fn sigmoid(z: &Array2<f32>) -> Array2<f32> {
            z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
        }

        fn relu(z: &Array2<f32>) -> Array2<f32> {
            z.mapv(|x| x.max(0.0))
        }

        let mut activations = HashMap::new();
        let mut a = input.clone(); // 输入的维度应为 (12288, N)

        for l in 1..self.layers.len() {
            let weight_key = format!("W{}", l);
            let bias_key = format!("b{}", l);
            let w = self.parameters.get(&weight_key).unwrap();
            let b = self.parameters.get(&bias_key).unwrap();

            // 确保矩阵乘法维度匹配
            let z = w.dot(&a) + b; // w 的维度应为 (输出节点数, 输入节点数)，a 的维度应为 (输入节点数, N)
            println!("Layer {}: z shape = {:?}", l, z.dim()); // 打印每层的 z 维度
            if l == self.layers.len() - 1 {
                // 输出层使用 sigmoid 激活函数
                a = sigmoid(&z);
            } else {
                // 隐藏层使用 ReLU 激活
                a = relu(&z);
            }
            println!("Layer {}: a shape = {:?}", l, a.dim()); // 打印每层的 a 维度
            activations.insert(format!("A{}", l), a.clone());
        }
        (activations, a)
    }

    // 优化：在计算成本时引入 L2 正则化
    // 交叉熵损失函数
    pub fn compute_cost(&self, predictions: &Array2<f32>, labels: &Array2<f32>) -> f32 {
        let m = labels.len_of(Axis(1)) as f32;
        let cross_entropy_cost = -1.0 / m
            * (labels * predictions.mapv(|x| x.ln())
                + (1.0 - labels) * (1.0 - predictions).mapv(|x| x.ln()))
            .sum();

        let l2_regularization_cost: f32 = (1..self.layers.len())
            .map(|l| {
                let w = self.parameters.get(&format!("W{}", l)).unwrap();
                w.mapv(|x| x.powi(2)).sum()
            })
            .sum();

        cross_entropy_cost + (0.01 / (2.0 * m)) * l2_regularization_cost
    }

    pub fn backward_propagation(
        &self,
        activations: &HashMap<String, Array2<f32>>,
        input: &Array2<f32>,
        labels: &Array2<f32>,
    ) -> HashMap<String, Array2<f32>> {
        let mut grads = HashMap::new();
        let m = input.len_of(Axis(1)) as f32;

        // 确保 labels 的维度与输出层的激活值维度匹配
        let labels = labels.to_owned(); // 克隆 labels 以匹配维度
        let a_last = activations
            .get(&format!("A{}", self.layers.len() - 1))
            .unwrap();
        let mut dz = a_last - &labels; // 假设最后一层是激活层

        // 打印初始 dz 的维度
        println!("Initial dz shape = {:?}", dz.dim());

        for l in (1..self.layers.len()).rev() {
            let a_prev_key = if l == 1 {
                "A0".to_string()
            } else {
                format!("A{}", l - 1)
            };
            let a_prev = if l == 1 {
                input
            } else {
                activations.get(&a_prev_key).unwrap()
            };

            // 打印 a_prev 的维度
            println!("Layer {}: a_prev shape = {:?}", l, a_prev.dim());

            let dw = 1.0 / m * dz.dot(&a_prev.t());
            let db = 1.0 / m * dz.sum_axis(Axis(1)).insert_axis(Axis(1));

            // 打印 dw 和 db 的维度
            println!("Layer {}: dw shape = {:?}", l, dw.dim());
            println!("Layer {}: db shape = {:?}", l, db.dim());

            grads.insert(format!("dW{}", l), dw);
            grads.insert(format!("db{}", l), db);

            if l > 1 {
                let w = self.parameters.get(&format!("W{}", l)).unwrap();
                dz = w.t().dot(&dz)
                    * activations.get(&format!("A{}", l - 1)).unwrap().mapv(|x| {
                        if x > 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    });

                // 打印 dz 的维度
                println!("Layer {}: dz shape = {:?}", l, dz.dim());
            }
        }
        grads
    }

    pub fn update_parameters(
        &mut self,
        grads: &HashMap<String, Array2<f32>>,
        optimizer: &mut dyn Optimizer,
    ) {
        optimizer.update(&mut self.parameters, grads, self.learning_rate);
    }

    pub fn train(
        &mut self,
        input: &Array2<f32>,
        labels: &Array2<f32>,
        epochs: usize,
        batch_size: usize,
        test_input: &Array2<f32>,
        test_labels: &Array2<f32>,
        optimizer: &mut dyn Optimizer,
    ) {
        let m = input.len_of(Axis(1));
        let num_batches = m / batch_size;
        let mut cost = 0.0;

        for epoch in 0..epochs {
            if epoch % 500 == 0 && epoch != 0 {
                self.learning_rate *= 0.9;
            }

            for i in 0..num_batches {
                let start = i * batch_size;
                let end = start + batch_size;
                let batch_input = input.slice(s![.., start..end]).to_owned();
                let batch_labels = labels.slice(s![.., start..end]).to_owned();

                let (activations, predictions) = self.forward_propagation(&batch_input);
                cost = self.compute_cost(&predictions, &batch_labels);
                let grads = self.backward_propagation(&activations, &batch_input, &batch_labels);
                self.update_parameters(&grads, optimizer);
            }

            if epoch % 100 == 0 {
                let test_predictions = self.predict(test_input);
                let test_cost = self.compute_cost(&test_predictions, test_labels);
                println!("Epoch {}: Cost: {} - Test Cost: {}", epoch, cost, test_cost);
            }
        }
    }
}
