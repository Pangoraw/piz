pub struct SearchWindow {
    shown: bool,
    query: String,
    results: Vec<&'static str>,
}

impl Default for SearchWindow {
    fn default() -> Self {
        Self {
            shown: false,
            query: String::new(),
            results: vec!["paragon", "self-supervised"],
        }
    }
}

impl SearchWindow {
    pub fn render(&mut self, ctx: &egui::Context) {
        if !self.shown {
            return;
        }

        egui::Window::new("search_window")
            .resizable(false)
            .show(ctx, |ui| {
                ui.text_edit_singleline(&mut self.query);
                for result in &self.results {
                    if !result.to_lowercase().contains(&self.query) {
                        continue;
                    }
                    ui.label(*result);
                }
            });
    }
}
