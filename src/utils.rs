use super::*;

pub fn dataframe_from_csv(file_path: PathBuf) -> Result<(DataFrame, DataFrame), Box<dyn Error>> {
    let data = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(file_path))?
        .finish()?;

    let training_dataset = data.drop("y")?;
    let training_labels = data.select(["y"])?;

    // 检查数据集和标签的维度
    println!("Training dataset shape: {:?}", training_dataset.shape());
    println!("Training labels shape: {:?}", training_labels.shape());

    Ok((training_dataset, training_labels))
}

pub fn array_from_dataframe(df: &DataFrame) -> Array2<f32> {
    df.to_ndarray::<Float32Type>(IndexOrder::Fortran)
        .unwrap()
        .reversed_axes()
}

// 加载测试数据集
pub fn load_test_data(file_path: PathBuf) -> Result<(DataFrame, DataFrame), Box<dyn Error>> {
    let data = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(file_path))?
        .finish()?;

    let test_dataset = data.drop("y")?;
    let test_labels = data.select(["y"])?;

    println!("Test dataset shape: {:?}", test_dataset.shape());
    println!("Test labels shape: {:?}", test_labels.shape());

    Ok((test_dataset, test_labels))
}
