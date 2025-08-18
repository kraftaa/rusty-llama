use std::fs::File;
use std::io::{BufRead, BufReader};
use anyhow::{Result, anyhow};

pub fn preprocess_forecast_input(path: &str) -> Result<Vec<f64>> {
    let file = File::open(path).map_err(|e| anyhow!("Failed to open file {}: {}", path, e))?;
    let reader = BufReader::new(file);

    let mut values = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let val: f64 = line.trim()
            .parse()
            .map_err(|e| anyhow!("Failed to parse float from '{}': {}", line, e))?;
        values.push(val);
    }

    Ok(values)
}