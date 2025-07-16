use ndarray::Array2;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};

struct DynamicsModel {
    session: Session,
}

impl DynamicsModel {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(DynamicsModel { session })
    }

    pub fn predict(&mut self, input: &[f32; 8]) -> Result<[f32; 6], Box<dyn std::error::Error>> {
        let input_array = Array2::from_shape_vec((1, 8), input.to_vec())?;

        let outputs = self
            .session
            .run(ort::inputs![TensorRef::from_array_view(&input_array)?])?;
        let output_tensor = outputs["output"].try_extract_array::<f32>()?;

        let mut result = [0.0; 6];
        for i in 0..6 {
            result[i] = output_tensor[[0, i]];
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const MODEL_PATH: &str = "models/dynamics_model.onnx";

    fn ensure_model_exists() {
        assert!(
            Path::new(MODEL_PATH).exists(),
            "Model file not found at {}.",
            MODEL_PATH
        );
    }

    #[test]
    fn test_dynamics_model_initialization() {
        ensure_model_exists();

        let model = DynamicsModel::new(MODEL_PATH);
        assert!(model.is_ok(), "Failed to initialize DynamicsModel");
    }

    #[test]
    fn test_dynamics_model_inference() {
        ensure_model_exists();

        let mut model = DynamicsModel::new(MODEL_PATH).expect("Failed to initialize model");
        let input = [0.0; 8];
        let result = model.predict(&input);

        assert!(result.is_ok(), "Prediction failed");
        let output = result.unwrap();

        println!("Model output: {:?}", output);
        assert_eq!(output.len(), 6, "Unexpected output dimension");

        // Verify all outputs are finite
        for (i, &value) in output.iter().enumerate() {
            assert!(
                value.is_finite(),
                "Output at index {} is not finite: {}",
                i,
                value
            );
        }
    }
}
