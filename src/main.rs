use std::io::Write;

use mupdf::Colorspace;
use wgpu::{include_wgsl, util::DeviceExt};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

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
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

struct QuadRenderer {
    texture_width: u32,
    texture_height: u32,
    quad_width: u32,
    quad_height: u32,
    pub changed: bool,
}

impl QuadRenderer {
    pub fn update(
        &mut self,
        texture_width: u32,
        texture_height: u32,
        quad_width: u32,
        quad_height: u32,
    ) -> bool {
        let mut changed = false;
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
        let texture_ratio = 0.5 * self.texture_width as f32 / self.texture_height as f32;
        let quad_ratio = self.quad_width as f32 / self.quad_height as f32;

        let texture_crop = texture_ratio / quad_ratio;

        if texture_crop > 1. {
            let quad_y = 1.0 - 2.0 * (1.0 / texture_crop);
            return [
                Vertex {
                    position: [-1.0, 1.0, 0.0],
                    tex_coords: [0.0, 0.0],
                }, // A
                Vertex {
                    position: [1.0, 1.0, 0.0],
                    tex_coords: [1.0, 0.0],
                }, // B
                Vertex {
                    position: [1.0, quad_y, 0.0],
                    tex_coords: [1.0, 1.0],
                }, // C
                Vertex {
                    position: [-1.0, quad_y, 0.0],
                    tex_coords: [0.0, 1.0],
                }, // D
            ];
        } else {
            return [
                Vertex {
                    position: [-1.0, 1.0, 0.0],
                    tex_coords: [0.0, 0.0],
                }, // A
                Vertex {
                    position: [1.0, 1.0, 0.0],
                    tex_coords: [1.0, 0.0],
                }, // B
                Vertex {
                    position: [1.0, -1.0, 0.0],
                    tex_coords: [1.0, texture_crop],
                }, // C
                Vertex {
                    position: [-1.0, -1.0, 0.0],
                    tex_coords: [0.0, texture_crop],
                }, // D
            ];
        }
    }
}

//                        A  C  B  A  D  C
const INDICES: &[u16] = &[0, 2, 1, 0, 3, 2];

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    texture: Option<wgpu::Texture>,
    renderer: Option<QuadRenderer>,

    color_r: f64,
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
        let num_indices = INDICES.len() as u32;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            bind_group: None,
            bind_group_layout: texture_bind_group_layout,
            texture: None,
            renderer: None,
            color_r: 0.3,
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

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.color_r = position.y as f64 / self.size.height as f64;
                true
            }
            _ => false,
        }
    }

    fn update(&mut self, texture_height: u32, texture_width: u32, winwidth: u32, winheight: u32) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.update(texture_width, texture_height, winwidth, winheight);
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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
                            r: 0.8 + 0.2 * self.color_r,
                            g: 1.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            if let Some(bind_group) = &self.bind_group {
                render_pass.set_bind_group(0, &bind_group, &[]);
            }

            if let Some(renderer) = self.renderer.as_mut() {
                if renderer.changed {
                    let vertices = renderer.get_vertices();
                    self.queue.write_buffer(
                        &self.vertex_buffer,
                        0,
                        bytemuck::cast_slice(&vertices),
                    );
                    renderer.changed = false;
                }
            }

            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn create_texture(&mut self, pixmap: &mupdf::Pixmap, winwidth: u32, winheight: u32) {
        let texture_size = wgpu::Extent3d {
            width: pixmap.width(),
            height: pixmap.height(),
            depth_or_array_layers: 1,
        };
        if let None = &self.texture {
            self.texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: Some("Pixmap Texture"),
            }));
            self.renderer = Some(QuadRenderer {
                texture_height: texture_size.height,
                texture_width: texture_size.width,
                quad_width: winwidth,
                quad_height: winheight,
                changed: true,
            });
        }

        let diffuse_texture = self.texture.as_ref().unwrap();
        self.queue.write_texture(
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

        if let None = self.bind_group {
            let diffuse_texture_view =
                diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let diffuse_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

            let diffuse_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
}

struct RenderedPage {
    page: mupdf::Page,
    pub pixmap: mupdf::Pixmap,
    textpage: mupdf::TextPage,
}

impl RenderedPage {
    pub fn new(doc: &mupdf::Document, page_count: i32) -> Result<Self, mupdf::Error> {
        let page = doc.load_page(page_count)?;

        let scale = 1.5;
        let mat = mupdf::Matrix::new_scale(scale, scale);

        let pixmap = page.to_pixmap(&mat, &Colorspace::device_rgb(), 1., false)?;
        let textpage = page.to_text_page(mupdf::TextPageOptions::BLOCK_TEXT)?;
        Ok(Self {
            page,
            pixmap,
            textpage,
        })
    }
}

async fn run() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let exe_name = &args[0];
    let filename = if args.len() != 2 {
        "/home/paul/Downloads/remotesensing-1853970.pdf"
    } else {
        &args[1]
    };

    let doc = mupdf::Document::open(filename).unwrap();

    let mut page_count = 0;
    let total_page_count = doc.page_count().unwrap() as usize;

    let mut pages = vec![
        RenderedPage::new(&doc, 0).unwrap(),
        RenderedPage::new(&doc, 1).unwrap(),
    ];
    let page = &pages[page_count];

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(format!("{} - {}", filename, exe_name))
        .with_inner_size(winit::dpi::PhysicalSize::new(
            page.pixmap.width(),
            page.pixmap.height(),
        ))
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await;

    for block in page.textpage.blocks() {
        for line in block.lines() {
            println!("Rect = {:?}", line.bounds());
            for char in line.chars() {
                print!("{}", char.char().unwrap());
            }
            println!("");
        }
    }

    // let page = doc.load_page(page_count).unwrap();

    // let scale = 7.0;
    // let mat = mupdf::Matrix::new_scale(scale, scale);

    // let pixmap = page
    //     .to_pixmap(&mat, &Colorspace::device_rgb(), 1., false)
    //     .unwrap();
    let winsize = window.inner_size();
    state.create_texture(&page.pixmap, winsize.width, winsize.height);

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
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
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Left),
                                ..
                            },
                        ..
                    } if page_count > 0 => {
                        page_count = (page_count - 1).max(0);

                        let page = &pages[page_count];
                        let winsize = window.inner_size();
                        state.create_texture(&page.pixmap, winsize.width, winsize.height);
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Right),
                                ..
                            },
                        ..
                    } => {
                        page_count = (page_count + 1).min(total_page_count - 1);

                        if pages.len() == page_count {
                            pages.push(RenderedPage::new(&doc, pages.len() as i32).unwrap());
                        }

                        let page = &pages[page_count];
                        let winsize = window.inner_size();
                        state.create_texture(&page.pixmap, winsize.width, winsize.height);
                    }
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
            state.update(
                pages[0].pixmap.width(),
                pages[0].pixmap.height(),
                winsize.width,
                winsize.height,
            );
            match state.render() {
                Ok(_) => {}

                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
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
    // let ctx = mupdf::Context::default();

    // to_ppm(&pixmap, "output.ppm").unwrap();
    // let mut file = std::fs::File::create("output.png").unwrap();
    // pixmap.write_to(&mut file, mupdf::ImageFormat::PNG).unwrap();

    pollster::block_on(run());
}

fn _to_ppm(pixmap: &mupdf::Pixmap, filepath: &str) -> Result<(), std::io::Error> {
    let mut file = std::fs::File::create(filepath)?;
    file.write(b"P3\n")?;
    file.write_fmt(format_args!("{} {} 255\n", pixmap.width(), pixmap.height()))?;

    let pixels = pixmap.samples();
    let stride = pixmap.stride() as usize;

    let mut y: usize = 0;
    while y < pixmap.height() as usize {
        let mut x: usize = 0;
        let new_pos = y * stride;
        let row_pixels = &pixels[new_pos..new_pos + stride];
        while x < pixmap.n() as usize * pixmap.width() as usize {
            if pixmap.n() == 4 && row_pixels[x + 3] == 0 {
                // TODO: Use the real formula.
                // Hacky rendering of alpha values but hey
                file.write(b"255 255 255\n")?;
            } else {
                file.write_fmt(format_args!(
                    "{} {} {}\n",
                    row_pixels[x],
                    row_pixels[x + 1],
                    row_pixels[x + 2],
                ))?;
            }
            x += pixmap.n() as usize;
        }
        y += 1;
    }

    Ok(())
}
