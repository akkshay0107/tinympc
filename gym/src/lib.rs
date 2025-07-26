use base::constants::*;
use base::world::World;
use pyo3::prelude::*;

#[pyclass]
struct PyEnvironment {
    world: World,
}

#[pymethods]
impl PyEnvironment {
    fn reset(&self) -> PyResult<[f32; 6]> {
        let ini_state = [0.0; 6];
        Ok(ini_state)
    }

    fn step(&self, action: [f32; 2]) -> PyResult<([f32; 6], f32, bool)> {
        let next_state = [0.0; 6];
        let reward = 0.0;
        let done = false;
        Ok((next_state, reward, done))
    }
}

#[pymodule]
fn gym(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnvironment>()?;
    Ok(())
}
