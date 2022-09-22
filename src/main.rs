use egui_winit::winit;
use egui_winit::winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use wgpu::{include_wgsl, util::DeviceExt};

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
    /// to relative quad space.
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
}

unsafe impl bytemuck::Pod for QuadVertex {}
unsafe impl bytemuck::Zeroable for QuadVertex {}

impl QuadVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];

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
    pub fn get_vertices(&self, renderer: &QuadRenderer) -> [QuadVertex; 4] {
        let (x0, y0) = renderer.from_texture(self.x, self.y).to_vertex_space();
        let (x1, y1) = renderer
            .from_texture(self.x + self.width, self.y + self.height)
            .to_vertex_space();

        [
            QuadVertex { position: [x0, y0] }, // A
            QuadVertex { position: [x1, y0] }, // B
            QuadVertex { position: [x1, y1] }, // C
            QuadVertex { position: [x0, y1] }, // D
        ]
    }
}

struct BlocksRenderPipeline {
    capacity: usize,

    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,

    quads: Vec<Quad>,
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
        }
    }

    fn render<'a, 'b>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_pass: &'b mut wgpu::RenderPass<'a>,
        renderer: &QuadRenderer,
    ) {
        render_pass.set_pipeline(&self.render_pipeline);

        // TODO: Move this to an instance buffer
        let vertices: Vec<QuadVertex> = self
            .quads
            .iter()
            .map(|q| q.get_vertices(renderer))
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

    fn find_hovering<I, T>(&self, it: I, point: Point) -> Option<T>
    where
        I: Iterator<Item = T>,
    {
        let quads = &self.quads;

        let x = point.x;
        let y = point.y;

        for (item, quad) in std::iter::zip(it, quads) {
            if quad.x < x && x < quad.x + quad.width && quad.y < y && y < quad.y + quad.height {
                return Some(item);
            }
        }

        None
    }

    fn hovers_quad(&self, point: Point) -> bool {
        return self.find_hovering(std::iter::repeat(()), point).is_some();
    }

    fn clear_blocks(&mut self) {
        self.quads.clear();
    }

    fn add_block(&mut self, quad: Quad) {
        self.quads.push(quad);
    }
}

struct Point {
    x: f32,
    y: f32,
}

impl Point {
    fn contained_in(&self, quad: mupdf::Quad) -> bool {
        quad.ul.x < self.x && quad.ul.y < self.y && quad.lr.x > self.x && quad.lr.y > self.y
    }

    fn to_vertex_space(&self) -> (f32, f32) {
        let x = 2.0 * self.x - 1.0;
        let y = 1.0 - 2.0 * self.y;
        (x, y)
    }
}

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

    fn render<'a, 'b>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_pass: &'b mut wgpu::RenderPass<'a>,
        renderer: &QuadRenderer,
    ) {
        self.block_render_pipeline
            .render(device, queue, render_pass, renderer);
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

        let page_render_pipeline = PageRenderPipeline {
            texture: None,
            render_pipeline,
            vertex_buffer,
            index_buffer,

            bind_group_layout: texture_bind_group_layout,
            bind_group: None,

            // TODO: Impl default::Default()
            renderer: QuadRenderer {
                x: 0.,
                y: 0.,
                texture_width: 0,
                texture_height: 0,
                quad_width: 0,
                quad_height: 0,
                changed: true,
            },
        };

        return page_render_pipeline;
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
    pub fn render<'a, 'b>(
        &'a self,
        queue: &wgpu::Queue,
        render_pass: &'b mut wgpu::RenderPass<'a>,
    ) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group.as_ref().unwrap(), &[]);

        if self.renderer.changed {
            let vertices = self.renderer.get_vertices();
            queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }

        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
    }

    fn from_pos_to_page(&self, pos: winit::dpi::PhysicalPosition<f64>) -> Point {
        let point = self.renderer.from_abs_quad(pos.x as f32, pos.y as f32);
        return self.renderer.from_quad(point.x, point.y);
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    page_render_pipeline: PageRenderPipeline,
    page_render_pipeline2: PageRenderPipeline,

    // TODO: Should be in the page
    block_render_pipeline: BlocksRenderPipeline,
    line_render_pipeline: BlocksRenderPipeline,
    link_render_pipeline: BlocksRenderPipeline,

    render_blocks: bool,
    render_lines: bool,
}

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
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

        let block_render_pipeline = BlocksRenderPipeline::new(&device, 10, &config, &shader);
        let line_render_pipeline = BlocksRenderPipeline::new(&device, 10, &config, &shader);
        let link_render_pipeline = BlocksRenderPipeline::new(&device, 10, &config, &shader);

        let page_render_pipeline = PageRenderPipeline::new(&device, &config, &shader);
        let page_render_pipeline2 = PageRenderPipeline::new(&device, &config, &shader);

        Self {
            surface,
            device,
            queue,
            config,
            size,

            page_render_pipeline,
            page_render_pipeline2,

            block_render_pipeline,
            line_render_pipeline,
            link_render_pipeline,

            render_blocks: false,
            render_lines: false,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn update(
        &mut self,
        x: f32,
        y: f32,
        texture_width: u32,
        texture_height: u32,
        winwidth: u32,
        winheight: u32,
    ) {
        self.page_render_pipeline.renderer.update(
            x,
            y,
            texture_width,
            texture_height,
            winwidth,
            winheight,
        );
        self.page_render_pipeline2.renderer.update(
            x,
            y + texture_height as f32,
            texture_width,
            texture_height,
            winwidth,
            winheight,
        );
    }

    fn render(
        &mut self,
        rp: &mut egui_wgpu::renderer::RenderPass,
        primitives: Vec<egui::epaint::ClippedPrimitive>,
        textures: egui::TexturesDelta,
    ) -> Result<(), wgpu::SurfaceError> {
        let quad_width = self.page_render_pipeline.renderer.quad_width;
        let quad_height = self.page_render_pipeline.renderer.quad_height;

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
                            r: 0.8,
                            g: 1.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            self.page_render_pipeline
                .render(&self.queue, &mut render_pass);
            self.page_render_pipeline2
                .render(&self.queue, &mut render_pass);
            if self.render_blocks {
                self.block_render_pipeline.render(
                    &self.device,
                    &self.queue,
                    &mut render_pass,
                    &self.page_render_pipeline.renderer,
                );
            }
            if self.render_lines {
                self.line_render_pipeline.render(
                    &self.device,
                    &self.queue,
                    &mut render_pass,
                    &self.page_render_pipeline.renderer,
                );
            }
            self.link_render_pipeline.render(
                &self.device,
                &self.queue,
                &mut render_pass,
                &self.page_render_pipeline.renderer,
            );

            rp.execute_with_renderpass(&mut render_pass, &primitives, &descriptor);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        self.page_render_pipeline.renderer.changed = false;

        Ok(())
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
            .hovers_quad(self.page_render_pipeline.from_pos_to_page(pos));
    }

    fn hovers_line(&self, pos: winit::dpi::PhysicalPosition<f64>) -> bool {
        return self
            .line_render_pipeline
            .hovers_quad(self.page_render_pipeline.from_pos_to_page(pos));
    }

    fn create_texture(
        &mut self,
        pixmap: &mupdf::Pixmap,
        winwidth: u32,
        winheight: u32,
        force: bool,
        first: bool,
    ) {
        if first {
            self.page_render_pipeline.create_texture(
                &self.device,
                &self.queue,
                pixmap,
                winwidth,
                winheight,
                force,
            );
        } else {
            self.page_render_pipeline2.create_texture(
                &self.device,
                &self.queue,
                pixmap,
                winwidth,
                winheight,
                force,
            );
        }
    }
}

struct RenderedPage {
    page: mupdf::Page,
    pixmap: mupdf::Pixmap,
    textpage: mupdf::TextPage,
}

impl RenderedPage {
    pub fn new(doc: &mupdf::Document, page_count: i32) -> Result<Self, mupdf::Error> {
        let page = doc.load_page(page_count)?;

        let mat = mupdf::Matrix::new_scale(1., 1.);
        let pixmap = page.to_pixmap(&mat, &mupdf::Colorspace::device_rgb(), 1., false)?;
        let textpage = page.to_text_page(mupdf::TextPageOptions::BLOCK_TEXT)?;
        Ok(Self {
            page,
            pixmap,
            textpage,
        })
    }

    fn rerender(&mut self, scale_factor: f32) -> Result<(), mupdf::Error> {
        let mat = mupdf::Matrix::new_scale(scale_factor, scale_factor);
        self.pixmap = self
            .page
            .to_pixmap(&mat, &mupdf::Colorspace::device_rgb(), 1., false)?;
        Ok(())
    }
}

fn run() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    tracing::debug!("args = {:?}", &args);

    let exe_name = &args[0];
    let filename = if args.len() != 2 {
        String::from("/home/paul/Downloads/remotesensing-1853970.pdf")
    } else {
        args.last().take().unwrap().to_string()
    };
    let prettyname = {
        let path = std::path::Path::new(&filename);
        String::from(path.file_name().unwrap().to_str().unwrap())
    };

    let doc = mupdf::Document::open(&filename).unwrap();

    let mut page_count = 0;
    let total_page_count = doc.page_count().unwrap() as usize;

    let mut ctx = mupdf::context::Context::get();
    ctx.set_text_aa_level(8);

    let mut pages = vec![RenderedPage::new(&doc, 0).unwrap()];
    if total_page_count > 1 {
        pages.push(RenderedPage::new(&doc, 1).unwrap());
    }
    let page = &mut pages[page_count];
    let bounds = page.page.bounds().unwrap();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(format!("{} - {}", prettyname, exe_name))
        .with_inner_size(winit::dpi::LogicalSize::new(
            bounds.width(),
            bounds.height(),
        ))
        .build(&event_loop)
        .unwrap();

    let mut state = pollster::block_on(State::new(&window));
    state.highlight_links(page).unwrap();
    state.highlight_blocks(page).unwrap();

    let scale_factor = window.scale_factor();
    page.rerender(scale_factor as f32).unwrap();

    let winsize = window.inner_size();
    state.create_texture(&page.pixmap, winsize.width, winsize.height, false, true);

    let page2 = &pages[1];
    state.create_texture(&page2.pixmap, winsize.width, winsize.height, false, false);

    let mut egui_state = egui_winit::State::new(&event_loop);
    let mut ctx = egui::Context::default();
    let mut rp =
        egui_wgpu::renderer::RenderPass::new(&state.device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);

    let mut cursor: Option<egui::CursorIcon> = None;
    let mut cursor_position = winit::dpi::PhysicalPosition { x: 0., y: 0. };

    let mut x = 0.;
    let mut y = 0.;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !egui_state.on_event(&mut ctx, &event) {
                match event {
                    WindowEvent::CursorMoved { position, .. } => {
                        if state.hovers_link(*position) {
                            cursor = Some(egui::CursorIcon::PointingHand);
                        } else if state.hovers_line(*position) {
                            cursor = Some(egui::CursorIcon::Text);
                        } else {
                            cursor = None;
                        }
                        cursor_position = *position;
                    }
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::MouseWheel {
                        delta: winit::event::MouseScrollDelta::LineDelta(xd, yd),
                        ..
                    } => {
                        x += xd * 10.;
                        y += yd * 10.;
                    }
                    WindowEvent::MouseInput {
                        state: winit::event::ElementState::Pressed,
                        ..
                    } => {
                        let point = state.page_render_pipeline.from_pos_to_page(cursor_position);
                        if let Some(link) = state
                            .link_render_pipeline
                            .find_hovering(pages[page_count].page.links().unwrap(), point)
                        {
                            if let Some(uri) = link.uri.strip_prefix("#page=") {
                                if let Some((page_number, _)) = uri.split_once('&') {
                                    let page_number =
                                        usize::from_str_radix(page_number, 10).unwrap();
                                    page_count = page_number; // Navigate to page
                                }
                            } else {
                                dbg!(link.uri);
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
                                        | VirtualKeyCode::B
                                        | VirtualKeyCode::L),
                                    ),
                                ..
                            },
                        ..
                    } => match keycode {
                        VirtualKeyCode::Left if page_count > 0 => {
                            page_count = (page_count - 1).max(0);

                            let page = &pages[page_count];
                            let winsize = window.inner_size();
                            state.highlight_links(page).unwrap();
                            state.highlight_blocks(page).unwrap();
                            state.create_texture(
                                &page.pixmap,
                                winsize.width,
                                winsize.height,
                                false,
                                false,
                            );
                        }
                        VirtualKeyCode::Right => {
                            page_count = (page_count + 1).min(total_page_count - 1);

                            if pages.len() == page_count {
                                pages.push(RenderedPage::new(&doc, pages.len() as i32).unwrap());
                            }

                            let page = &pages[page_count];
                            let winsize = window.inner_size();
                            state.highlight_links(page).unwrap();
                            state.highlight_blocks(page).unwrap();
                            state.create_texture(
                                &page.pixmap,
                                winsize.width,
                                winsize.height,
                                false,
                                false,
                            );
                        }
                        VirtualKeyCode::B => {
                            state.render_blocks = !state.render_blocks;
                        }
                        VirtualKeyCode::L => {
                            state.render_lines = !state.render_lines;
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
            if (winsize.width).abs_diff(pages[page_count].pixmap.width()) >= 1 {
                let page_width = pages[page_count].page.bounds().unwrap().width();
                let new_scale_factor = winsize.width as f32 / page_width;
                pages[page_count].rerender(new_scale_factor as f32).unwrap();
                state.create_texture(
                    &pages[page_count].pixmap,
                    winsize.width,
                    winsize.height,
                    true,
                    false,
                );
            }

            let mut output = ctx.run(egui_state.take_egui_input(&window), |ctx| {
                egui::TopBottomPanel::bottom("bottom_panel").show(&ctx, |ui| {
                    ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "{}/{}",
                                page_count + 1,
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
            });
            if let Some(cursor) = cursor {
                if !ctx.is_pointer_over_area() {
                    output.platform_output.cursor_icon = cursor;
                }
            }
            egui_state.handle_platform_output(&window, &ctx, output.platform_output);

            let primitives = ctx.tessellate(output.shapes);
            let textures = output.textures_delta;

            state.resize(winsize);
            state.update(
                x,
                y,
                pages[page_count].pixmap.width(),
                pages[page_count].pixmap.height(),
                winsize.width,
                winsize.height,
            );

            match state.render(&mut rp, primitives, textures) {
                Ok(_) => {}

                Err(wgpu::SurfaceError::Lost) => state.resize(winsize),
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
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
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    run();
}
