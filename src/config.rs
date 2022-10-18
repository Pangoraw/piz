use serde_derive::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub(crate) zotero_dir: Option<PathBuf>,
    pub(crate) background_color: (f64, f64, f64, f64),
}

impl Default for Config {
    fn default() -> Self {
        Self {
            zotero_dir: None,
            background_color: (1., 1., 1., 1.),
        }
    }
}

impl Config {
    pub fn from_file(path: PathBuf) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;

        Ok(config)
    }
}
