struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

struct QuadVertex {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct QuadVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn quad_vs_main(
    model: QuadVertex,
) -> QuadVertexOutput {
    var out: QuadVertexOutput;
    out.clip_position = vec4(model.position, 1.0, 1.0);
    out.color = model.color;
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

// Dark mode implementation is taken directly from sioyek and translated to wgsl - https://github.com/ahrm/sioyek
// http://gamedev.stackexchange.com/questions/59797/glsl-shader-change-hue-saturation-brightness
fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(), vec3<f32>(1.0, 1.0, 1.0)), c.y);
}

@fragment
fn fs_main_dark(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let hsv_color = rgb2hsv(color.xyz);

    let hsv_lightness = hsv_color.b;
    let rgb_lightness = min(min(color.r, color.g), color.b);
    let new_lightness = (rgb_lightness + hsv_lightness) / 2.0;

    let new_color = hsv2rgb(vec3(hsv_color.xy, 1.0 - new_lightness));
    return vec4(new_color, color.w);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}

@fragment
fn quad_fs_main(in: QuadVertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
