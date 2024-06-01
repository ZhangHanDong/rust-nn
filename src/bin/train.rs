use ndarray::prelude::*;
use rust_nn::*;
use std::path::Path;

fn main() {
    // 定义神经网络的层结构和学习率
    // 在深度神经网络中，vec![12288, 128, 64, 32, 16, 1] 代表的是每一层的神经元（节点）数量。
    // 具体地说：
    //      1.	`12288`：输入层的节点数量。这通常与输入数据的特征维度相对应。
    //          输入数据是大小为 64 x 64 的 3 通道矩阵（图像），展开后得到 12288（即 64 * 64 * 3）。
    //      2.	`128`：第一层隐藏层的节点数量。该层有 128 个神经元。
    //      3.	`64`：第二层隐藏层的节点数量。该层有 64 个神经元。
    //      4.	`32`：第三层隐藏层的节点数量。该层有 32 个神经元。
    //      5.	`16`：第四层隐藏层的节点数量。该层有 16 个神经元。
    //      6.	`1`：输出层的节点数量。
    //         因为这是一个二分类问题（识别猫和非猫），输出层只有一个节点，其输出通常经过一个 sigmoid 激活函数，表示图像是猫的概率。
    // 我们也可以设置一个小型的网络结构：vec![12288, 3, 5, 10, 1];
    // let neural_network_layers: Vec<usize> = vec![12288, 128, 64, 32, 16, 1];
    let neural_network_layers: Vec<usize> = vec![12288, 3, 5, 10, 1];
    let learning_rate = 0.001;
    let epochs = 2000;
    let batch_size = 64;

    // 读取命令行参数
    let optimizer_name = "sgd";

    // 加载训练数据集
    let (training_dataset, training_labels) =
        dataframe_from_csv("datasets/training_set.csv".into()).unwrap();
    let training_set_array = array_from_dataframe(&training_dataset);
    let training_labels_array = array_from_dataframe(&training_labels);

    // 加载测试数据集
    let (test_dataset, test_labels) = load_test_data("datasets/test_set.csv".into()).unwrap();
    let test_set_array = array_from_dataframe(&test_dataset);
    let test_labels_array = array_from_dataframe(&test_labels);

    // 初始化和训练模型
    let mut model = DeepNeuralNetwork::new(neural_network_layers.clone(), learning_rate);
    model.initialize_parameters();

    let mut optimizer: Box<dyn Optimizer> = match optimizer_name {
        "adam" => Box::new(Adam::new(0.9, 0.999, 1e-8)),
        "sgd" => Box::new(SGD),
        _ => unreachable!(),
    };

    model.train(
        &training_set_array,
        &training_labels_array,
        epochs,
        batch_size,
        &test_set_array,
        &test_labels_array,
        optimizer.as_mut(),
    );

    // 保存训练后的模型参数
    let save_path = Path::new("trained_model.json");
    model
        .save_parameters(save_path)
        .expect("Failed to save model parameters");

    // 加载训练后的模型参数
    let mut loaded_model = DeepNeuralNetwork::new(neural_network_layers.clone(), learning_rate);
    loaded_model
        .load_parameters(save_path)
        .expect("Failed to load model parameters");

    // 使用加载后的模型进行预测
    let prediction = loaded_model.predict(&test_set_array);
    println!("Prediction: {:?}", prediction);
}
