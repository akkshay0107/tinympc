use ndarray::ArrayD;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use std::error::Error;

pub struct PolicyNet {
    session: Session,
}

impl PolicyNet {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    /// Forward pass. Returns (mean, std) vectors.
    pub fn forward(
        &mut self,
        input: Vec<f32>,
        input_shape: Vec<usize>,
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
        let input_tensor = Value::from_array(ArrayD::from_shape_vec(input_shape, input)?)?;
        let inputs = ort::inputs!["input" => input_tensor]; // No `?`
        let outputs = self.session.run(inputs)?;

        if outputs.len() != 2 {
            return Err("Expected two outputs: mean and std".into());
        }

        let (_, mean_slice) = outputs[0].try_extract_tensor::<f32>()?;
        let mean = mean_slice.to_vec();

        let (_, std_slice) = outputs[1].try_extract_tensor::<f32>()?;
        let std = std_slice.to_vec();

        Ok((mean, std))
    }

    /// Get raw action from forward pass and convert it into a valid control
    pub fn get_action(
        &mut self,
        input: Vec<f32>,
        input_shape: Vec<usize>,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let (mean, _) = self.forward(input, input_shape).unwrap(); // std not required during inference
        let action: Vec<f32> = mean.into_iter().map(|x| x.tanh()).collect();
        Ok(action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_net_forward() {
        let mut net = PolicyNet::new("../python/models/policy_net.onnx").unwrap();
        let obs_dim = 6;
        let act_dim = 2;
        let input_shape = vec![1, obs_dim]; // batch size 1
        let input = vec![0.0; obs_dim];

        let (mean, std) = net.forward(input, input_shape).unwrap();
        println!("{:?}", mean);
        assert_eq!(mean.len(), act_dim);
        assert_eq!(std.len(), act_dim);
    }

    #[test]
    fn test_policy_net_get_action() {
        let mut net = PolicyNet::new("../python/models/policy_net.onnx").unwrap();
        let obs_dim = 6;
        let act_dim = 2;
        let input_shape = vec![1, obs_dim]; // batch size 1
        let input = vec![0.0; obs_dim];

        let action = net.get_action(input, input_shape).unwrap();
        println!("{:?}", action);
        assert_eq!(action.len(), act_dim);
    }
}
