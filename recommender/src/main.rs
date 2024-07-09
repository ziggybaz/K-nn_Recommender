mod knn

use knn::{classify_with_dynamic_k_value};

fn main() {
    let mut data: Vec<DataPoint> = vec![];

    normalize(&mut data);

    let pca_data = apply_pca(&data, 2);

    let test_point = DataPoint {};

    let predicted_label = classify_with_dynamic_k_value(&pca_data, &test_point);

    println!("The predicted label is: {}", predicted_label);
}
