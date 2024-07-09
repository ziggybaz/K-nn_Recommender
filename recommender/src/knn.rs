extern crate nalgebra as na;

use na::{DMatrix};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct DataPoint {
    pub features: Vec<f64>,
    pub label: Option<String>,
}

pub fn normalize(data: &mut [DataPoint]) {
    let n_features = data[0].features.len();

    for i in 0..n_features {
        let mut minimum_value = f64::INFINITY;
        let mut maximum_value = f64::NEG_INFINITY;

        for point in data.iter() {
            minimum_value = minimum_value.min(point.features[i]);
            maximum_value = maximum_value.max(point.features[i]);
        }

        for point in data.iter_mut() {
            point.features[i] = (point.features[i] - minimum_value) / (maximum_value - minimum_value);
        }
    }
}

pub fn apply_pca(data: &[DataPoint], n_components: usize) -> Vec<DataPoint> {
    let n_samples = data.len();
    let n_features = data[0].features.len();

    let mut matrix = DMatrix::from_element(n_samples, n_features, 0.0);

    for (i, point) in data.iter().enumerate() {
        for (j, &feature) in point.features.iter().enumerate() {
            matrix[(i, j)] = feature;
        }
    }

    let cov_matrix = (matrix.transpose() * &matrix) / (n_samples as f64 - 1.0);
    let eig = cov_matrix.symmetric_eigen();

    let mut pca_data = vec![];
    for i in 0..n_samples {
        let mut transformed_features = vec![];
        for j in 0..n_components { transformed_features.push(eig.eigenvectors.column(j).dot(&matrix.row(i).transpose())) }
    
    pca_data.push(DataPoint {
        features:transformed_features,
        label: data[i].label.clone(),
    });
    }
    pca_data
}

pub fn euclidean_distance(a:&DataPoint, b: &DataPoint) -> f64 {
    a.features.iter()
        .zip(&b.features)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub fn calculate_local_density(data: &[DataPoint], point:&DataPoint) -> f64 {
    let mut distances: Vec<f64> = data.iter()
        .map(|other| euclidean_distance(point, other))
        .collect();

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let k = 5;
    distances.iter().take(k).sum::<f64>() / (k as f64)
}

pub fn determine_dynamic_k_value(data: &[DataPoint], point: &DataPoint) -> usize {
    let local_density = calculate_local_density(data, point);
    let k_minimum_value = 1;
    let k_maximum_value = 15;

    let optimal_k_value = ((local_density * (k_maximum_value as f64 - k_minimum_value as f64)).ceil() as usize).min(k_maximum_value).max(k_minimum_value);

    optimal_k_value
}

pub fn classify_with_dynamic_k_value(data: &[DataPoint], test_point: &DataPoint) -> String {
    let k = determine_dynamic_k_value(data, test_point);

    let mut distances: Vec<(f64, String)> = data.iter()
        .filter_map(|point| {
            point.label.as_ref().map(|label| (euclidean_distance(point, test_point), label.clone()))
        })
        .collect();

    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut class_counts = HashMap::new();
    for (_, label) in distances.iter().take(k) {
        *class_counts.entry(label.clone()).or_insert(0) += 1;
    }

    class_counts.into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(label, _)| label)
        .unwrap_or_else(|| "Error".to_string())
}
