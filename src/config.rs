use serde_derive::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub(crate) struct Config {
    pub(crate) zotero_dir: Option<PathBuf>,
    pub(crate) background_color: (f64, f64, f64, f64),
    pub(crate) dark_background_color: (f64, f64, f64, f64),
    pub(crate) dark_mode: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            zotero_dir: None,
            // Solarize light
            background_color: (0.93, 0.91, 0.84, 1.0),
            // Solarize dark
            dark_background_color: (0.108, 0.108, 0.150, 1.0),
            dark_mode: false,
        }
    }
}

impl Config {
    pub(crate) fn from_file(path: PathBuf) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;

        Ok(config)
    }
}
