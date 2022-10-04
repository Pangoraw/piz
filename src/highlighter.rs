struct TextHighlighter {
    text_page: mupdf::TextPage,
    block_render_pipeline: BlocksRenderPipeline,
    anchor: Option<Point>,
    head: Option<Point>,
}

impl TextHighlighter {
    fn new(page: mupdf::TextPage, block_render_pipeline: BlocksRenderPipeline) -> Self {
        Self {
            text_page: page,
            block_render_pipeline,
            anchor: None,
            head: None,
        }
    }

    fn start_selection(&mut self, pos: winit::dpi::PhysicalPosition<f64>) {
        // TODO: transform this to page space
        self.anchor = Some(Point {
            x: pos.x as f32,
            y: pos.y as f32,
        });
        self.head = None;
    }

    fn move_cursor(&mut self, cursor: winit::dpi::PhysicalPosition<f64>) {
        if let None = self.anchor {
            return;
        }

        self.head = Some(Point {
            x: cursor.x as f32,
            y: cursor.y as f32,
        });
    }

    fn compute_quads(&mut self) {
        self.block_render_pipeline.clear_blocks();

        enum SelectState {
            SeekStart,
            SeekEnd,
        }
        let mut state = SelectState::SeekStart;

        /*
        match (self.head.as_ref(), self.anchor.as_ref()) {
            (Some(head), Some(anchor)) => {
                for block in self.text_page.blocks() {
                    for line in block.lines() {
                        let bounds = line.bounds();
                        let line_start = bounds.origin();
                        let x = line_start.x;
                        let y = line_start.y;

                        for char in line.chars() {
                            let endpoint =
                                head.contained_in(char.quad()) && anchor.contained_in(char.quad());
                            if let SelectState::SeekStart = state && endpoint {
                                state = SelectState::SeekEnd
                            } else if let SelectState::SeekEnd = state && endpoint {

                            }
                        }
                        self.block_render_pipeline.add_block(Quad {
                            x,
                            y,
                            width: 0.05,
                            height: 0.05,
                        });
                    }
                }
            }
            _ => {}
        };
        */
    }

    fn render<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_pass: &mut wgpu::RenderPass<'a>,
        renderer: &QuadRenderer,
    ) {
        self.block_render_pipeline
            .render(device, queue, render_pass, renderer);
    }
}


