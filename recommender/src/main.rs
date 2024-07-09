mod knn;

use knn::{normalize, apply_pca, classify_with_dynamic_k_value};

fn main() {
    let mut data: Vec<DataPoint> = vec![
        DataPoint { features: vec![0.5, 0.2, 0.1], label: Some("A".to_string()) },
        DataPoint { features: vec![0.9, 0.7, 0.6], label: Some("B".to_string()) },
        DataPoint { features: vec![0.4, 0.5, 0.4], label: Some("A".to_string()) },
        DataPoint { features: vec![0.8, 0.6, 0.5], label: Some("B".to_string()) },
        DataPoint { features: vec![0.3, 0.4, 0.3], label: Some("A".to_string()) },
        DataPoint { features: vec![0.7, 0.8, 0.7], label: Some("B".to_string()) },
    ];

    normalize(&mut data);

    let pca_data = apply_pca(&data, 2);

    let test_point = DataPoint {
        features: vec![0.55, 0.25],
        label: None,
    };

    let predicted_label = classify_with_dynamic_k_value(&pca_data, &test_point);

    println!("The predicted label is: {}", predicted_label);
}
