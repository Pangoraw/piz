use egui_winit::winit;
use egui_winit::winit::{
    event::*,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};
use wgpu::{include_wgsl, util::DeviceExt};

mod search_bar;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[derive(Debug)]
struct QuadRenderer {
    x: f32,
    y: f32,
    texture_width: u32,
    texture_height: u32,
    quad_width: u32,
    quad_height: u32,
    changed: bool,
}

impl Default for QuadRenderer {
    fn default() -> Self {
        Self {
            x: 0.,
            y: 0.,
            texture_width: 0,
            texture_height: 0,
            quad_width: 0,
            quad_height: 0,
            changed: true,
        }
    }
}

impl QuadRenderer {
    pub fn update(
        &mut self,
        x: f32,
        y: f32,
        texture_width: u32,
        texture_height: u32,
        quad_width: u32,
        quad_height: u32,
    ) -> bool {
        let mut changed = self.changed;
        if x != self.x {
            self.x = x;
            changed = true;
        }
        if y != self.y {
            self.y = y;
            changed = true;
        }
        if texture_width != self.texture_width {
            self.texture_width = texture_width;
            changed = true;
        }
        if texture_height != self.texture_height {
            self.texture_height = texture_height;
            changed = true;
        }
        if quad_width != self.quad_width {
            self.quad_width = quad_width;
            changed = true;
        }
        if quad_height != self.quad_height {
            self.quad_height = quad_height;
            changed = true;
        }

        self.changed = changed;
        return changed;
    }

    /// Transforms coordinates from the texture in [0, 1]
    /// to the screen/quad coordinates frame in [0, 1].
    pub fn from_texture(&self, x: f32, y: f32) -> Point {
        let quad_x = self.x as f32 + x * self.texture_width as f32;
        let quad_y = self.y as f32 + y * self.texture_height as f32;

        Point {
            x: quad_x / self.quad_width as f32,
            y: quad_y / self.quad_height as f32,
        }
    }

    /// Transforms from the absolute screen/quad space [0, w/h]
    /// to relative quad space [0, 1].
    pub fn from_abs_quad(&self, x: f32, y: f32) -> Point {
        Point {
            x: x / self.quad_width as f32,
            y: y / self.quad_height as f32,
        }
    }

    /// Transforms coordinates from the quad on the screen in [0, 1]
    /// to the texture coordinates frame in [0, 1].
    pub fn from_quad(&self, x: f32, y: f32) -> Point {
        let text_x = x * self.quad_width as f32 - self.x as f32;
        let text_y = y * self.quad_height as f32 - self.y as f32;

        Point {
            x: text_x / self.texture_width as f32,
            y: text_y / self.texture_height as f32,
        }
    }

    /// Checks whether the point in texture position [0, 1] is in
    /// the texture.
    pub fn contains(&self, point: &Point) -> bool {
        1. <= point.x && point.x <= 0. && 1. <= point.y && point.y <= 0.
    }

    pub fn get_vertices(&self) -> [Vertex; 4] {
        //
        // position coordinates   texture coordinates
        // frame                  frame
        //
        //  A       1       B     0 ------ -----> 1
        //          ^             |
        //          |             |
        //          |             |
        // -1 <---- 0 ----> 1     |
        //          |             |
        //          |             |
        //          v             v
        //  D      -1       C     1
        //
        let (x0, y0) = self.from_texture(0., 0.).to_vertex_space();
        let (x1, y1) = self.from_texture(1., 1.).to_vertex_space();

        return [
            Vertex {
                position: [x0, y0, 0.0],
                tex_coords: [0.0, 0.0],
            }, // A
            Vertex {
                position: [x1, y0, 0.0],
                tex_coords: [1.0, 0.0],
            }, // B
            Vertex {
                position: [x1, y1, 0.0],
                tex_coords: [1.0, 1.0],
            }, // C
            Vertex {
                position: [x0, y1, 0.0],
                tex_coords: [0.0, 1.0],
            }, // D
        ];
    }
}

//                        A  C  B  A  D  C
const INDICES: &[u16] = &[0, 2, 1, 0, 3, 2];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct QuadVertex {
    position: [f32; 2],
    color: [f32; 4],
}

unsafe impl bytemuck::Pod for QuadVertex {}
unsafe impl bytemuck::Zeroable for QuadVertex {}

impl QuadVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[derive(Debug)]
struct Quad {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl Default for Quad {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
        }
    }
}

impl Quad {
    pub fn get_vertices(&self, renderer: &QuadRenderer, color: &egui::Rgba) -> [QuadVertex; 4] {
        let (x0, y0) = renderer.from_texture(self.x, self.y).to_vertex_space();
        let (x1, y1) = renderer
            .from_texture(self.x + self.width, self.y + self.height)
            .to_vertex_space();

        let color = [color.r(), color.g(), color.b(), color.a()];
        [
            QuadVertex {
                position: [x0, y0],
                color,
            }, // A
            QuadVertex {
                position: [x1, y0],
                color,
            }, // B
            QuadVertex {
                position: [x1, y1],
                color,
            }, // C
            QuadVertex {
                position: [x0, y1],
                color,
            }, // D
        ]
    }
}

struct BlocksRenderPipeline {
    capacity: usize,

    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,

    quads: Vec<Quad>,
    normal_color: egui::Rgba,
}

impl BlocksRenderPipeline {
    fn new(
        device: &wgpu::Device,
        n_blocks: usize,
        config: &wgpu::SurfaceConfiguration,
        shader: &wgpu::ShaderModule,
    ) -> Self {
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Block Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "quad_vs_main",
                buffers: &[QuadVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "quad_fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::OVER,
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block Render Vertex Buffer"),
            mapped_at_creation: false,
            size: (n_blocks * 4 * std::mem::size_of::<QuadVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Block Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        let quads = Vec::with_capacity(n_blocks);

        Self {
            capacity: n_blocks,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            quads,
            normal_color: egui::Rgba::from_rgba_unmultiplied(0.8, 0.2, 0.3, 0.3),
        }
    }

    fn render<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_pass: &mut wgpu::RenderPass<'a>,
        renderer: &QuadRenderer,
    ) {
        render_pass.set_pipeline(&self.render_pipeline);

        // TODO: Move this to an instance buffer
        let vertices: Vec<QuadVertex> = self
            .quads
            .iter()
            .map(|q| q.get_vertices(renderer, &self.normal_color))
            .flatten()
            .collect();
        let indices: Vec<u16> = (0..4 * self.quads.len())
            .step_by(4)
            .map(|i| i as u16)
            .map(|i| [i + 0, i + 2, i + 1, i + 0, i + 3, i + 2])
            .flatten()
            .collect();

        if self.capacity <= self.quads.len() {
            self.vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Block Render Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Block Render Vertex Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
            queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
        }

        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
    }

    /// Returns the first item zipped with the Quad that contains the given Point
    /// in texture space [0, 1].
    fn find_hovering<I, T>(&self, it: I, point: &Point) -> Option<T>
    where
        I: Iterator<Item = T>,
    {
        // If the point is not in valid texture space, assume that no quad will
        // intercept it.
        if point.x < 0. || point.x > 1. || point.y < 0. || point.y > 1. {
            return None;
        }

        let quads = &self.quads;

        for (item, quad) in std::iter::zip(it, quads) {
            if point.contained_in(quad) {
                return Some(item);
            }
        }

        None
    }

    /// Returns whether or not any quad from this render pipeline contains the
    /// given point in texture space [0, 1].
    fn hovers_quad(&self, point: &Point) -> bool {
        return self.find_hovering(std::iter::repeat(()), point).is_some();
    }

    fn clear_blocks(&mut self) {
        self.quads.clear();
    }

    fn add_block(&mut self, quad: Quad) {
        self.quads.push(quad);
    }
}

#[derive(Clone, Debug)]
struct Point {
    x: f32,
    y: f32,
}

impl Point {
    fn contained_in(&self, quad: &Quad) -> bool {
        quad.x + quad.width >= self.x
            && self.x >= quad.x
            && quad.y + quad.height >= self.y
            && self.y >= quad.y
    }

    fn to_vertex_space(&self) -> (f32, f32) {
        let x = 2.0 * self.x - 1.0;
        let y = 1.0 - 2.0 * self.y;
        (x, y)
    }
}

struct PageRenderPipeline {
    texture: Option<wgpu::Texture>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,

    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,

    renderer: QuadRenderer,

    block_render_pipeline: BlocksRenderPipeline,
    line_render_pipeline: BlocksRenderPipeline,
    link_render_pipeline: BlocksRenderPipeline,
    search_render_pipeline: BlocksRenderPipeline,
}

impl PageRenderPipeline {
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        shader: &wgpu::ShaderModule,
    ) -> Self {
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("Pixmap Texture Bind Group Layout"),
            });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            mapped_at_creation: false,
            size: (4 * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let block_render_pipeline = BlocksRenderPipeline::new(&device, 0, &config, &shader);
        let line_render_pipeline = BlocksRenderPipeline::new(&device, 0, &config, &shader);
        let link_render_pipeline = BlocksRenderPipeline::new(&device, 0, &config, &shader);
        let search_render_pipeline = BlocksRenderPipeline::new(&device, 0, &config, &shader);

        let page_render_pipeline = PageRenderPipeline {
            texture: None,
            render_pipeline,
            vertex_buffer,
            index_buffer,

            bind_group_layout: texture_bind_group_layout,
            bind_group: None,

            renderer: QuadRenderer::default(),

            block_render_pipeline,
            line_render_pipeline,
            link_render_pipeline,
            search_render_pipeline,
        };

        return page_render_pipeline;
    }

    /// Returns wether or not parts of the rendered texture are visible on the screen.
    fn is_visible(&self) -> bool {
        let ul = self.renderer.from_texture(0., 0.);
        let br = self.renderer.from_texture(1., 1.);

        ((ul.y >= 0. && ul.y <= 1.) || (br.y >= 0. && br.y <= 1.) || (ul.y < 0. && br.y > 1.))
            && ((ul.x >= 0. && ul.x <= 1.)
                || (br.x >= 0. && br.x <= 1.)
                || (ul.x < 0. && br.x > 1.))
    }

    fn highlight_blocks(&mut self, page: &RenderedPage) -> Result<(), mupdf::Error> {
        self.block_render_pipeline.clear_blocks();
        self.line_render_pipeline.clear_blocks();

        let page_bounds = page.page.bounds()?;

        let blocks = page.textpage.blocks();
        for block in blocks {
            for line in block.lines() {
                let rect = line.bounds();

                self.line_render_pipeline.add_block(Quad {
                    x: rect.x0 / page_bounds.width(),
                    y: rect.y0 / page_bounds.height(),
                    width: (rect.x1 - rect.x0) / page_bounds.width(),
                    height: (rect.y1 - rect.y0) / page_bounds.height(),
                });
            }
            let rect = block.bounds();
            self.block_render_pipeline.add_block(Quad {
                x: rect.x0 / page_bounds.width(),
                y: rect.y0 / page_bounds.height(),
                width: (rect.x1 - rect.x0) / page_bounds.width(),
                height: (rect.y1 - rect.y0) / page_bounds.height(),
            });
        }

        Ok(())
    }

    fn highlight_links(&mut self, page: &RenderedPage) -> Result<(), mupdf::Error> {
        self.link_render_pipeline.clear_blocks();

        let page_bounds = page.page.bounds()?;
        for link in page.page.links()? {
            let rect = link.bounds;
            self.link_render_pipeline.add_block(Quad {
                x: rect.x0 / page_bounds.width(),
                y: rect.y0 / page_bounds.height(),
                width: (rect.x1 - rect.x0) / page_bounds.width(),
                height: (rect.y1 - rect.y0) / page_bounds.height(),
            });
        }

        Ok(())
    }

    fn hovers_link(&self, pos: winit::dpi::PhysicalPosition<f64>) -> bool {
        return self
            .link_render_pipeline
            .hovers_quad(&self.from_pos_to_page(pos));
    }

    fn hovers_line(&self, pos: winit::dpi::PhysicalPosition<f64>) -> bool {
        return self
            .line_render_pipeline
            .hovers_quad(&self.from_pos_to_page(pos));
    }

    pub fn create_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pixmap: &mupdf::Pixmap,
        winwidth: u32,
        winheight: u32,
        force: bool,
    ) {
        let texture_size = wgpu::Extent3d {
            width: pixmap.width(),
            height: pixmap.height(),
            depth_or_array_layers: 1,
        };
        if self.texture.is_none() || force {
            self.texture = Some(device.create_texture(&wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: Some("Pixmap Texture"),
            }));
            self.renderer
                .update(0., 0., pixmap.width(), pixmap.height(), winwidth, winheight);
        }

        let diffuse_texture = self.texture.as_ref().unwrap();
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixmap.samples(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * pixmap.width()),
                rows_per_image: std::num::NonZeroU32::new(pixmap.height()),
            },
            texture_size,
        );

        if self.bind_group.is_none() || force {
            let diffuse_texture_view =
                diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

            let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                    },
                ],
                label: Some("Pixmap Texture Bind Group"),
            });

            self.bind_group = Some(diffuse_bind_group);
        }
    }
    pub fn render<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_pass: &mut wgpu::RenderPass<'a>,
        render_blocks: bool,
        render_lines: bool,
        render_links: bool,
    ) {
        if !self.is_visible() {
            return;
        }

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group.as_ref().unwrap(), &[]);

        if self.renderer.changed {
            let vertices = self.renderer.get_vertices();
            queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }

        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);

        if render_blocks {
            self.block_render_pipeline
                .render(device, queue, render_pass, &self.renderer);
        }
        if render_lines {
            self.line_render_pipeline
                .render(device, queue, render_pass, &self.renderer);
        }
        if render_links {
            self.link_render_pipeline
                .render(device, queue, render_pass, &self.renderer);
        }
        self.search_render_pipeline
            .render(device, queue, render_pass, &self.renderer);
    }

    fn from_pos_to_page(&self, pos: winit::dpi::PhysicalPosition<f64>) -> Point {
        let point = self.renderer.from_abs_quad(pos.x as f32, pos.y as f32);
        return self.renderer.from_quad(point.x, point.y);
    }
}

struct RenderedPage {
    page: mupdf::Page,
    display_list: mupdf::DisplayList,
    pixmap: mupdf::Pixmap,
    textpage: mupdf::TextPage,
}

impl RenderedPage {
    pub fn new(doc: &mupdf::Document, page_count: i32) -> Result<Self, mupdf::Error> {
        let page = doc.load_page(page_count)?;

        let display_list = page.to_display_list(false)?;
        let textpage = display_list.to_text_page(mupdf::TextPageOptions::BLOCK_TEXT)?;

        // TODO: Create pixmap with right size right away.
        let mat = mupdf::Matrix::new_scale(1., 1.);
        let pixmap = display_list.to_pixmap(&mat, &mupdf::Colorspace::device_rgb(), true)?;

        Ok(Self {
            page,
            display_list,
            pixmap,
            textpage,
        })
    }

    fn rerender(&mut self, scale_factor: f32) -> Result<(), mupdf::Error> {
        let mat = mupdf::Matrix::new_scale(scale_factor, scale_factor);
        self.pixmap = self
            .display_list
            .to_pixmap(&mat, &mupdf::Colorspace::device_rgb(), true)?;
        Ok(())
    }
}

struct Page {
    render_pipeline: PageRenderPipeline,
    page: RenderedPage,
}

impl Page {
    fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        shader: &wgpu::ShaderModule,
        doc: &mupdf::Document,
        page_count: i32,
    ) -> Result<Self, mupdf::Error> {
        let render_pipeline = PageRenderPipeline::new(&device, &config, &shader);
        let rendered_page = RenderedPage::new(doc, page_count)?;
        Ok(Self {
            render_pipeline,
            page: rendered_page,
        })
    }

    fn create_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        winwidth: u32,
        winheight: u32,
        force: bool,
    ) {
        if self.physical_width() != winwidth {
            self.page
                .rerender(winwidth as f32 / self.width().unwrap())
                .unwrap();
            self.render_pipeline.create_texture(
                device,
                queue,
                &self.page.pixmap,
                winwidth,
                winheight,
                force,
            );
        }
    }

    const MAX_SEARCH_RESULTS: u32 = 10;
    fn search(&mut self, query: &str) -> Result<usize, mupdf::Error> {
        self.render_pipeline.search_render_pipeline.clear_blocks();

        if query.is_empty() {
            return Ok(0);
        }

        let page_bounds = self.page.page.bounds()?;
        let results = self.page.page.search(query, Self::MAX_SEARCH_RESULTS)?;
        for result in &results {
            self.render_pipeline.search_render_pipeline.add_block(Quad {
                x: result.ul.x / page_bounds.width(),
                y: result.ul.y / page_bounds.height(),
                width: (result.lr.x - result.ul.x) / page_bounds.width(),
                height: (result.lr.y - result.ul.y) / page_bounds.height(),
            });
        }

        Ok(results.len())
    }

    fn highlight_blocks(&mut self) -> Result<(), mupdf::Error> {
        self.render_pipeline.highlight_blocks(&self.page)?;
        Ok(())
    }

    pub fn highlight_links(&mut self) -> Result<(), mupdf::Error> {
        self.render_pipeline.highlight_links(&self.page)?;
        Ok(())
    }

    pub fn hovers_link(&self, position: winit::dpi::PhysicalPosition<f64>) -> bool {
        self.render_pipeline.hovers_link(position)
    }

    pub fn hovers_line(&self, position: winit::dpi::PhysicalPosition<f64>) -> bool {
        self.render_pipeline.hovers_line(position)
    }

    fn find_hovering_link(&self, point: &Point) -> Option<mupdf::Link> {
        match self.page.page.links() {
            Ok(links) => self
                .render_pipeline
                .link_render_pipeline
                .find_hovering(links, point),
            Err(_) => None,
        }
    }

    fn width(&self) -> Result<f32, mupdf::Error> {
        let rect = self.page.page.bounds()?;
        Ok(rect.width())
    }

    fn height(&self) -> Result<f32, mupdf::Error> {
        let rect = self.page.page.bounds()?;
        Ok(rect.height())
    }

    fn physical_width(&self) -> u32 {
        self.render_pipeline.renderer.texture_width
    }

    fn physical_height(&self) -> u32 {
        self.render_pipeline.renderer.texture_height
    }

    fn update(
        &mut self,
        x: f32,
        y: f32,
        winwidth: u32,
        winheight: u32,
    ) -> Result<(), mupdf::Error> {
        let quad_w = self.render_pipeline.renderer.quad_width;
        let quad_h = self.render_pipeline.renderer.quad_height;

        self.render_pipeline.renderer.update(
            x,
            y,
            self.page.pixmap.width(),
            self.page.pixmap.height(),
            winwidth,
            winheight,
        );

        if quad_w != winwidth || quad_h != winheight {
            self.page.rerender(winwidth as f32 / self.width()?)?;
        }

        Ok(())
    }

    fn render<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_pass: &mut wgpu::RenderPass<'a>,
        render_blocks: bool,
        render_lines: bool,
        render_links: bool,
    ) {
        self.render_pipeline.render(
            device,
            queue,
            render_pass,
            render_blocks,
            render_lines,
            render_links,
        );
    }

    fn is_visible(&self) -> bool {
        self.render_pipeline.is_visible()
    }
}

struct State {
    adapter: wgpu::Adapter,
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    shader: wgpu::ShaderModule,

    size: winit::dpi::PhysicalSize<u32>,

    pages: Vec<Page>,
    position: Point,
    page_count: usize,

    render_blocks: bool,
    render_lines: bool,
    render_links: bool,
    render_nav_bar: bool,
    show_debug: bool,

    cached_query: String,
}

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: Some("device"),
                },
                None,
            )
            .await
            .unwrap();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(include_wgsl!("shader.wgsl"));

        Self {
            adapter,
            surface,
            device,
            queue,
            config,
            size,
            shader,

            pages: vec![],
            position: Point { x: 0., y: 0. },
            page_count: 0,

            render_blocks: false,
            render_lines: false,
            render_links: true,
            render_nav_bar: true,
            show_debug: false,

            cached_query: String::new(),
        }
    }

    fn add_page(&mut self, doc: &mupdf::Document, page_count: i32) -> Result<(), mupdf::Error> {
        if page_count >= doc.page_count()? {
            return Ok(());
        }

        let mut page = Page::new(&self.device, &self.config, &self.shader, doc, page_count)?;
        page.highlight_blocks()?;
        page.highlight_links()?;
        page.create_texture(
            &self.device,
            &self.queue,
            self.size.width,
            self.size.height,
            false,
        );
        self.pages.push(page);
        Ok(())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }

        // FIXME: hot path
        self.create_texture(new_size.width, new_size.height, true);
    }

    fn update(&mut self, doc: &mupdf::Document) -> Result<(), mupdf::Error> {
        let mut offset = 0.;
        for (i, page) in self.pages.iter_mut().enumerate() {
            page.update(
                self.position.x,
                self.position.y + offset,
                self.size.width,
                self.size.height,
            )?;

            if -self.position.y >= offset {
                self.page_count = i;
            }

            offset += page.physical_height() as f32;
        }

        // offset sums the physical height of all pages
        // if offset is now in frame, create a page.
        if -self.position.y < offset && -self.position.y + self.size.height as f32 > offset {
            self.add_page(doc, self.pages.len() as i32).unwrap();
        }

        Ok(())
    }

    fn render(
        &mut self,
        rp: &mut egui_wgpu::renderer::RenderPass,
        primitives: Vec<egui::epaint::ClippedPrimitive>,
        textures: egui::TexturesDelta,
    ) -> Result<(), wgpu::SurfaceError> {
        let quad_width = self.size.width;
        let quad_height = self.size.height;

        let descriptor = egui_wgpu::renderer::ScreenDescriptor {
            size_in_pixels: [quad_width, quad_height],
            pixels_per_point: 1.0, // What should go there ? window.scale_factor() ?
        };

        for (id, image_delta) in textures.set {
            rp.update_texture(&self.device, &self.queue, id, &image_delta);
            assert!(rp.texture(&id).is_some());
        }
        rp.update_buffers(&self.device, &self.queue, &primitives, &descriptor);

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Path"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 238. / 255., // Solarized: base1 RGBA(238, 232, 213, 1)
                            g: 232. / 255., // Solarized: base2 RGBA(253, 246, 227, 1)
                            b: 213. / 255.,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            for page in self.pages.iter_mut() {
                page.render(
                    &self.device,
                    &self.queue,
                    &mut render_pass,
                    self.render_blocks,
                    self.render_lines,
                    self.render_links,
                );
            }
            rp.execute_with_renderpass(&mut render_pass, &primitives, &descriptor);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        for page in self.pages.iter_mut() {
            page.render_pipeline.renderer.changed = false;
        }

        Ok(())
    }

    fn create_texture(&mut self, winwidth: u32, winheight: u32, force: bool) {
        for page in self.pages.iter_mut() {
            page.create_texture(&self.device, &self.queue, winwidth, winheight, force);
        }
    }

    fn hovers_link(&self, position: winit::dpi::PhysicalPosition<f64>) -> bool {
        self.pages
            .iter()
            .filter(|page| page.is_visible())
            .any(|page| page.hovers_link(position))
    }

    fn hovers_line(&self, position: winit::dpi::PhysicalPosition<f64>) -> bool {
        self.pages
            .iter()
            .filter(|page| page.is_visible())
            .any(|page| page.hovers_line(position))
    }

    fn find_hovering_link(&self, pos: winit::dpi::PhysicalPosition<f64>) -> Option<mupdf::Link> {
        for page in self.pages.iter().filter(|page| page.is_visible()) {
            let point = page.render_pipeline.from_pos_to_page(pos);

            match page.find_hovering_link(&point) {
                opt @ Some(_) => return opt,
                None => continue,
            }
        }
        None
    }

    fn navigate_to(
        &mut self,
        doc: &mupdf::Document,
        page_number: usize,
    ) -> Result<(), mupdf::Error> {
        if page_number >= doc.page_count()? as usize {
            return Ok(());
        }

        // Add missing pages
        while self.pages.len() <= page_number {
            self.add_page(doc, self.pages.len() as i32)?;
        }

        self.position.y = 0.;
        for page in self.pages.iter().take(page_number) {
            self.position.y -= page.physical_height() as f32;
        }
        self.page_count = page_number;

        Ok(())
    }

    /// Clears everything and render the document again and then moves at the previous position.
    fn rerender_document(&mut self, doc: &mupdf::Document) -> Result<(), mupdf::Error> {
        let page_count = self.page_count;
        let y = self.position.y;
        self.clear_pages();
        self.navigate_to(doc, page_count)?;
        self.position.y = y;
        Ok(())
    }

    fn search(&mut self, query: &str) -> Result<(), mupdf::Error> {
        if query == self.cached_query {
            return Ok(());
        }

        for page in self.pages.iter_mut() {
            page.search(query)?;
        }
        self.cached_query = query.to_string();

        Ok(())
    }

    fn clear_pages(&mut self) {
        self.pages = vec![];
        self.page_count = 0;
    }
}

const SCROLL_DELTA: f32 = 20.;

fn run() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    tracing::debug!("args = {:?}", &args);

    let exe_name = &args[0];
    if args.len() != 2 {
        Err::<(), &str>("Please provide the file name").unwrap();
    }
    let filename = args.last().take().unwrap().to_string();
    let mut prettyname = {
        let path = std::path::Path::new(&filename);
        String::from(path.file_name().unwrap().to_str().unwrap())
    };

    let mut doc = mupdf::Document::open(&filename).unwrap();
    let (winwidth, winheight) = {
        let first_page = doc.load_page(0).unwrap();
        let pixmap = first_page
            .to_pixmap(
                &mupdf::Matrix::IDENTITY,
                &mupdf::Colorspace::device_rgb(),
                1.,
                false,
            )
            .unwrap();
        (pixmap.width(), pixmap.height())
    };

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(format!("{} - {}", prettyname, exe_name))
        .with_inner_size(winit::dpi::LogicalSize::new(winwidth, winheight))
        .build(&event_loop)
        .unwrap();

    let mut state = pollster::block_on(State::new(&window));
    state.add_page(&doc, 0).unwrap();
    state.add_page(&doc, 1).unwrap();

    let mut egui_state = egui_winit::State::new(&event_loop);
    let mut ctx = egui::Context::default();

    let texture_format = state.surface.get_supported_formats(&state.adapter)[0];
    let mut rp = egui_wgpu::renderer::RenderPass::new(&state.device, texture_format, 1);

    let mut bar = search_bar::SearchWindow::default();

    let mut cursor: Option<egui::CursorIcon> = None;
    let mut cursor_position = winit::dpi::PhysicalPosition { x: 0., y: 0. };

    let mut last_render_time = std::time::Instant::now();
    let mut query = String::new();

    let mut outlines = doc.outlines().unwrap();
    let mut show_table_of_content = false;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !egui_state.on_event(&mut ctx, &event) {
                match event {
                    WindowEvent::CloseRequested => control_flow.set_exit(),
                    WindowEvent::CursorMoved { position, .. } => {
                        // TODO: Recompute this based on scrolling too.
                        if state.hovers_link(*position) {
                            cursor = Some(egui::CursorIcon::PointingHand);
                        } else if state.hovers_line(*position) {
                            cursor = Some(egui::CursorIcon::Text);
                        } else {
                            cursor = None;
                        }
                        cursor_position = *position;
                    }
                    WindowEvent::MouseWheel {
                        delta: winit::event::MouseScrollDelta::LineDelta(xd, yd),
                        ..
                    } => {
                        state.position.x += SCROLL_DELTA * xd;
                        state.position.y = (state.position.y + SCROLL_DELTA * yd).min(0.);
                    }
                    WindowEvent::MouseInput {
                        state: winit::event::ElementState::Pressed,
                        button: winit::event::MouseButton::Left,
                        ..
                    } => {
                        if let Some(link) = state.find_hovering_link(cursor_position) {
                            if let Some(uri) = link.uri.strip_prefix("#page=") {
                                if let Some((page_number, _)) = uri.split_once('&') {
                                    let page_number =
                                        usize::from_str_radix(page_number, 10).unwrap() - 1;
                                    if page_number < doc.page_count().unwrap() as usize {
                                        state.navigate_to(&doc, page_number).unwrap();
                                    }
                                }
                            } else {
                                webbrowser::open(&link.uri).unwrap();
                            }
                        }
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode:
                                    Some(
                                        keycode @ (VirtualKeyCode::Left
                                        | VirtualKeyCode::Right
                                        | VirtualKeyCode::Up
                                        | VirtualKeyCode::Down
                                        | VirtualKeyCode::D
                                        | VirtualKeyCode::R
                                        | VirtualKeyCode::T
                                        | VirtualKeyCode::Slash),
                                    ),
                                ..
                            },
                        ..
                    } => match keycode {
                        VirtualKeyCode::Left if state.page_count > 0 => {
                            state.navigate_to(&doc, state.page_count - 1).unwrap();
                        }
                        VirtualKeyCode::Right => {
                            state.navigate_to(&doc, state.page_count + 1).unwrap();
                        }
                        VirtualKeyCode::Down => {
                            state.position.y -= 20.;
                        }
                        VirtualKeyCode::Up => {
                            state.position.y = (state.position.y + 20.).min(0.);
                        }
                        VirtualKeyCode::D => {
                            state.show_debug = !state.show_debug;
                        }
                        VirtualKeyCode::R => {
                            doc = mupdf::Document::open(&filename).unwrap();
                            outlines = doc.outlines().unwrap();
                            state.rerender_document(&doc).unwrap();
                        }
                        VirtualKeyCode::T => {
                            show_table_of_content = !show_table_of_content;
                        }
                        VirtualKeyCode::Slash => {
                            bar.toggle_shown();
                        }
                        _ => {}
                    },
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            let winsize = window.inner_size();

            let mut should_refresh_doc = false;
            let mut output = ctx.run(egui_state.take_egui_input(&window), |ctx| {
                if state.render_nav_bar {
                    egui::TopBottomPanel::bottom("bottom_panel").show(&ctx, |ui| {
                        ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "{}/{}",
                                    state.page_count + 1,
                                    doc.page_count().unwrap()
                                ))
                                .size(20.),
                            );
                            ui.centered_and_justified(|ui| {
                                let mut job = egui::text::LayoutJob::single_section(
                                    prettyname.to_owned(),
                                    egui::text::TextFormat::simple(
                                        egui::FontId {
                                            size: 20.,
                                            family: egui::text::FontFamily::Monospace,
                                        },
                                        egui::Color32::GRAY,
                                    ),
                                );
                                job.wrap = egui::epaint::text::TextWrapping {
                                    max_rows: 1,
                                    overflow_character: Some('…'),
                                    ..Default::default()
                                };
                                ui.label(job);
                            });
                        });
                    });
                }

                bar.render(ctx);

                egui::Window::new("Table of Content")
                    .open(&mut show_table_of_content)
                    .show(&ctx, |ui| {
                        for outline in &outlines {
                            egui::CollapsingHeader::new(&outline.title).show(ui, |ui| {
                                for outline in &outline.down {
                                    ui.label(&outline.title);
                                }
                            });
                        }
                    });

                egui::Window::new("Debug Window")
                    .open(&mut state.show_debug)
                    .show(&ctx, |ui| {
                        ui.text_edit_singleline(&mut query);

                        should_refresh_doc = ui.button("Refresh doc").clicked();

                        ui.checkbox(&mut state.render_lines, "Render lines");
                        ui.checkbox(&mut state.render_blocks, "Render blocks");
                        ui.checkbox(&mut state.render_links, "Render links");
                        ui.checkbox(&mut state.render_nav_bar, "Render nav bar");

                        let elapsed = last_render_time.elapsed();
                        let fps = std::time::Duration::from_secs(1).as_nanos() / elapsed.as_nanos();
                        ui.monospace(format!("{:?}", elapsed));
                        ui.monospace(format!("{}fps", fps));
                        last_render_time = std::time::Instant::now();
                    });
            });
            if let Some(cursor) = cursor {
                if !ctx.is_pointer_over_area() {
                    output.platform_output.cursor_icon = cursor;
                }
            }
            egui_state.handle_platform_output(&window, &ctx, output.platform_output);
            if output.repaint_after.is_zero() {
                control_flow.set_poll();
            } else {
                control_flow.set_wait()
            }

            if should_refresh_doc || bar.has_file_to_open() {
                let filename = if bar.has_file_to_open() {
                    let path = bar.file_to_open().unwrap();
                    prettyname = path.file_name().unwrap().to_str().unwrap().to_string();

                    path.to_str().unwrap().to_string()
                } else {
                    filename.to_string()
                };

                doc = mupdf::Document::open(&filename).unwrap();
                outlines = doc.outlines().unwrap();
                state.rerender_document(&doc).unwrap();
            }

            let primitives = ctx.tessellate(output.shapes);
            let textures = output.textures_delta;

            state.resize(winsize);
            state.update(&doc).unwrap();
            state.search(&query).unwrap();

            match state.render(&mut rp, primitives, textures) {
                Ok(_) => {}

                Err(wgpu::SurfaceError::Lost) => state.resize(winsize),
                Err(wgpu::SurfaceError::OutOfMemory) => control_flow.set_exit(),
                Err(e) => println!("error: {:?}", e),
            }
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    })
}

fn main() {
    let home_dir = home::home_dir().unwrap();
    let log_file = std::fs::File::options()
        .append(true)
        .create(true)
        .open(home_dir.join(".local/share/piz.log"))
        .unwrap();

    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::TRACE)
        .with_writer(std::sync::Arc::new(log_file))
        .with_writer(std::io::stderr)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    run();
}
