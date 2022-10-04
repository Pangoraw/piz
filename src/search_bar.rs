use std::path::PathBuf;

#[derive(Debug, Clone)]
struct ZoteroFile {
    title: String,
    path: String,
    key: String,
}

impl ZoteroFile {
    fn path(&self) -> PathBuf {
        PathBuf::from(zotero_path())
            .join("storage")
            .join(&self.key)
            .join(&self.path)
    }
}

pub struct SearchWindow {
    shown: bool,
    was_just_shown: bool,
    query: String,
    results: Vec<ZoteroFile>,

    should_open_file: Option<usize>,
    currently_selected: Option<usize>,
}

fn zotero_path() -> PathBuf {
    match std::env::var("HOSTNAME") {
        Ok(s) => match s.as_str() {
            "grapefruit" => home::home_dir().unwrap().join("Zotero"),
            _ => home::home_dir()
                .unwrap()
                .join("snap/zotero-snap/common/Zotero"),
        },
        _ => home::home_dir()
            .unwrap()
            .join("snap/zotero-snap/common/Zotero"),
    }
}

impl Default for SearchWindow {
    fn default() -> Self {
        let connection = sqlite::Connection::open_with_flags(
            zotero_path().join("zotero.sqlite"),
            sqlite::OpenFlags::new().set_read_only(),
        )
        .unwrap();

        let cursor = connection
            .prepare(
                r#"
        SELECT value, itemAttachments.path, attachmentItems.key
        FROM items
        LEFT JOIN itemData, itemDataValues, itemAttachments, items attachmentItems
        WHERE itemData.itemID = items.itemID
            AND itemData.fieldID = 1
            AND itemData.valueID = itemDataValues.valueID
            AND itemAttachments.path LIKE 'storage:%'
            AND itemAttachments.parentItemID = items.itemID
            AND attachmentItems.itemID = itemAttachments.itemID
            AND itemAttachments.contentType = 'application/pdf';
        "#,
            )
            .unwrap()
            .into_cursor();

        let results: Vec<ZoteroFile> = cursor
            .filter_map(|f| match f {
                Ok(row) => {
                    let file = ZoteroFile {
                        title: row.get(0),
                        path: row
                            .get::<String, _>(1)
                            .strip_prefix("storage:")
                            .unwrap()
                            .to_string(),
                        key: row.get(2),
                    };
                    if file.path().exists() {
                        Some(file)
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();

        Self {
            shown: false,
            was_just_shown: false,
            query: String::new(),
            results,
            should_open_file: None,
            currently_selected: None,
        }
    }
}

impl SearchWindow {
    pub fn has_file_to_open(&self) -> bool {
        self.should_open_file.is_some()
    }

    pub fn file_to_open(&mut self) -> Option<PathBuf> {
        let res = self
            .should_open_file
            .map(|index| self.results[index].path());
        self.should_open_file = None;
        res
    }

    pub fn render(&mut self, ctx: &egui::Context) {
        if !self.shown {
            return;
        }

        egui::Window::new("Zotero Search")
            .open(&mut self.shown)
            .show(ctx, |ui| {
                let text = egui::TextEdit::singleline(&mut self.query)
                    .hint_text("Search...")
                    .cursor_at_end(true)
                    .desired_width(f32::INFINITY);

                let text = ui.add(text);

                if self.was_just_shown {
                    text.request_focus();
                    self.was_just_shown = false;
                }

                if self.query.trim().is_empty() {
                    return;
                }

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (i, (title, pos)) in
                            self.results.iter().enumerate().filter_map(|(i, result)| {
                                match result.title.to_lowercase().find(&self.query) {
                                    Some(pos) => Some((i, (&result.title, pos))),
                                    None => None,
                                }
                            })
                        {
                            // if self.currently_selected.is_none() {
                            //     self.currently_selected = Some(0);
                            // }

                            let (first_part, second_part) = title.split_at(pos);
                            let (second_part, third_part) = second_part.split_at(self.query.len());
                            let color = if let Some(idx) = self.currently_selected {
                                if idx == i {
                                    egui::Color32::LIGHT_GRAY
                                } else {
                                    egui::Color32::GRAY
                                }
                            } else {
                                egui::Color32::GRAY
                            };

                            let mut job = egui::text::LayoutJob::default();
                            job.append(
                                &first_part,
                                0.0,
                                egui::TextFormat::simple(
                                    egui::FontId {
                                        size: 25.,
                                        family: egui::text::FontFamily::Monospace,
                                    },
                                    color,
                                ),
                            );
                            job.append(
                                &second_part,
                                0.0,
                                egui::TextFormat::simple(
                                    egui::FontId {
                                        size: 25.,
                                        family: egui::text::FontFamily::Monospace,
                                    },
                                    egui::Color32::LIGHT_RED,
                                ),
                            );
                            job.append(
                                &third_part,
                                0.0,
                                egui::TextFormat::simple(
                                    egui::FontId {
                                        size: 25.,
                                        family: egui::text::FontFamily::Monospace,
                                    },
                                    color,
                                ),
                            );
                            job.wrap = egui::epaint::text::TextWrapping {
                                max_rows: 1,
                                overflow_character: Some('…'),
                                break_anywhere: true,
                                ..Default::default()
                            };

                            let button = ui.add(egui::Button::new(job).frame(false));
                            if button.hovered() {
                                ui.output().cursor_icon = egui::CursorIcon::PointingHand;
                            }
                            if button.clicked() {
                                self.should_open_file = Some(i);
                            }
                        }
                    });
            });
    }

    pub fn toggle_shown(&mut self) -> bool {
        self.shown = !self.shown;
        if self.shown {
            self.was_just_shown = true;
        }
        self.shown
    }
}
