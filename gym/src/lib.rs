pub mod gym_env;
use pyo3::prelude::*;

#[pyfunction]
fn test(a: u32) -> PyResult<u32> {
    Ok(a)
}

#[pymodule]
fn gym(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test, m)?)
}
