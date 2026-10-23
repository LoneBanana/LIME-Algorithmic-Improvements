// WebGL2 / OpenGL ES 3.0 renderer with a manim-style timeline, compiled to WebAssembly.
//
// The page builds a scene through the extern "C" API below (objects, tweens, camera)
// and the engine renders it every animation frame. manim concepts map as follows:
//   Create     -> tween REVEAL 0 -> 1   (geometry carries a reveal parameter)
//   FadeIn/Out -> tween ALPHA
//   Transform  -> tween MORPH 0 -> 1    (every vertex/instance has a morph target)
//   ambient camera rotation, camera moves -> camera tweens
// Higher dimensions are shown by sweeping a 4-D field through time (iso-surface of
// f(x0 + u a + v b + w c + t d)) and by a grand tour of d-dimensional data.
#include <GLES3/gl3.h>
#include <emscripten/emscripten.h>
#include <emscripten/html5.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "geometry.hpp"

namespace {

const char *kVS = R"(#version 300 es
layout(location=0) in vec3 aPosA;
layout(location=1) in vec3 aNrmA;
layout(location=2) in vec3 aPosB;
layout(location=3) in vec3 aNrmB;
layout(location=4) in vec4 aColor;
layout(location=5) in float aParam;
layout(location=6) in vec3 iCenterA;
layout(location=7) in vec3 iCenterB;
layout(location=8) in vec4 iColor;
layout(location=9) in float iRadius;
uniform mat4 uViewProj;
uniform float uMorph;
uniform int uInstanced;
uniform vec3 uOffset;
uniform float uScale;
uniform vec3 uScaleOrigin;
out vec3 vPos;
out vec3 vNrm;
out vec4 vColor;
out float vParam;
void main() {
  vec3 p = mix(aPosA, aPosB, uMorph);
  vec3 n = mix(aNrmA, aNrmB, uMorph);
  vec4 col = aColor;
  float param = aParam;
  if (uInstanced == 1) {
    p = mix(iCenterA, iCenterB, uMorph) + aPosA * iRadius;
    n = aNrmA;
    col = iColor;
    param = 0.0;
  }
  p = uScaleOrigin + (p - uScaleOrigin) * uScale + uOffset;
  vPos = p;
  vNrm = n;
  vColor = col;
  vParam = param;
  gl_Position = uViewProj * vec4(p, 1.0);
}
)";

const char *kFS = R"(#version 300 es
precision highp float;
in vec3 vPos;
in vec3 vNrm;
in vec4 vColor;
in float vParam;
uniform vec3 uEye;
uniform float uAlpha;
uniform float uReveal;
uniform vec4 uTint;
uniform float uUnlit;
uniform float uEmissive;
uniform int uClip;
uniform vec3 uClipLo;
uniform vec3 uClipHi;
out vec4 frag;
void main() {
  if (vParam > uReveal + 1e-4) discard;
  if (uClip == 1 && (any(lessThan(vPos, uClipLo)) || any(greaterThan(vPos, uClipHi)))) discard;
  vec3 n = normalize(vNrm + vec3(1e-7));
  vec3 v = normalize(uEye - vPos);
  if (dot(n, v) < 0.0) n = -n;
  vec3 base = vColor.rgb * uTint.rgb;
  vec3 l1 = normalize(vec3(0.45, -0.55, 0.75));
  vec3 l2 = normalize(vec3(-0.6, 0.45, 0.3));
  float diff = max(dot(n, l1), 0.0) * 0.78 + max(dot(n, l2), 0.0) * 0.22;
  vec3 h = normalize(l1 + v);
  float spec = pow(max(dot(n, h), 0.0), 48.0) * 0.32;
  float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.28;
  vec3 lit = base * (0.30 + 0.70 * diff) + vec3(spec) + base * rim;
  vec3 c = mix(lit, base, uUnlit) + base * uEmissive;
  frag = vec4(c, vColor.a * uTint.a * uAlpha);
}
)";

enum Prop { ALPHA = 0, REVEAL = 1, MORPH = 2, SCALE = 3, EMISSIVE = 4, TIME = 5, OFFSET_Z = 6, VISIBLE = 7 };
enum CamProp { THETA = 10, PHI = 11, DIST = 12, TARGET_Z = 13, ORTHO_SIZE = 14 };
enum Kind { STATIC = 0, ISO4D = 1, TOUR_POINTS = 2, TOUR_WALK = 3, TOUR_ARROWS = 4, TOUR_AXES = 5 };

struct Gpu {
    GLuint vao = 0, vbo = 0, ibo = 0, inst = 0;
    int nidx = 0, ninst = 0;
};

struct Object {
    int id = 0;
    Kind kind = STATIC;
    Gpu gpu;
    bool instanced = false, transparent = false, clip = false, pickable = false, unlit = false;
    float alpha = 1, reveal = 1, morph = 0, scale = 1, emissive = 0, time = 0, offset_z = 0, visible = 1;
    float last_time = -1e9f;
    Vec3 center, scale_origin;
    RGBA tint{1, 1, 1, 1};
    std::vector<float> inst_data;  // per instance: centreA(3) centreB(3) colour(4) radius(1)
    // ISO4D payload
    std::vector<float> fields;
    int nt = 0, nx = 0, ny = 0, nz = 0;
    float iso = 0.5f;
    Vec3 lo, hi;
    RGBA color{1, 1, 1, 1};
};

struct Tween {
    int target, prop;
    float from, to, start, dur;
    int easing;
    bool done = false;  // camera tweens: final value already applied
};

struct Camera {
    float theta = -1.0f, phi = 1.05f, dist = 13.0f, fov = 0.62f;
    Vec3 target{0, 0, 1.0f};
    bool ortho = false;
    float ortho_size = 4.0f;
};

struct TourState {
    bool active = false;
    Tour tour;
    int n = 0;
    std::vector<float> X;
    float t = 0, speed = 0.12f, scale = 1.0f;
    int points = -1, walk = -1, arrows = -1, axes = -1;
    std::vector<float> walk_rows, arrow_origin, arrow_vecs;
    int walk_m = 0, arrow_k = 0;
    RGBA walk_color{1, 1, 1, 1};
    std::vector<RGBA> arrow_colors;
    float axes_len = 2.5f;
    std::vector<Vec3> axis_tips;
    std::vector<float> frame;
};

EMSCRIPTEN_WEBGL_CONTEXT_HANDLE g_ctx = 0;
std::string g_selector;
GLuint g_prog = 0, g_sphere_vbo = 0, g_sphere_ibo = 0;
int g_sphere_nidx = 0;
std::map<std::string, GLint> g_uni;
std::map<int, Object> g_objects;
std::vector<Tween> g_tweens;
Camera g_cam;
TourState g_tour;
int g_next_id = 1;
int g_px_w = 800, g_px_h = 600;
float g_css_w = 800, g_css_h = 600;
double g_last_ms = 0;
float g_time = 0, g_speed = 1;
bool g_playing = true, g_paused = false;
float g_auto_rotate = 0.0f;
float g_bg[3] = {0.055f, 0.063f, 0.078f};
bool g_clip_on = false;
Vec3 g_clip_lo, g_clip_hi;
Mat4 g_view_proj = Mat4::identity();
Vec3 g_eye;

GLint uni(const char *name) {
    auto it = g_uni.find(name);
    if (it != g_uni.end()) return it->second;
    GLint loc = glGetUniformLocation(g_prog, name);
    g_uni[name] = loc;
    return loc;
}

GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::printf("shader error: %s\n", log);
    }
    return s;
}

void setup_mesh_attribs() {
    const GLsizei stride = sizeof(Vertex);
    const int sizes[6] = {3, 3, 3, 3, 4, 1};
    size_t offset = 0;
    for (int a = 0; a < 6; ++a) {
        glEnableVertexAttribArray(a);
        glVertexAttribPointer(a, sizes[a], GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void *>(offset));
        offset += sizes[a] * sizeof(float);
    }
}

void setup_instance_attribs() {
    const GLsizei stride = 11 * sizeof(float);
    const int sizes[4] = {3, 3, 4, 1};
    size_t offset = 0;
    for (int a = 0; a < 4; ++a) {
        glEnableVertexAttribArray(6 + a);
        glVertexAttribPointer(6 + a, sizes[a], GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void *>(offset));
        glVertexAttribDivisor(6 + a, 1);
        offset += sizes[a] * sizeof(float);
    }
}

void upload_mesh(Object &o, const MeshData &m) {
    if (!o.gpu.vao) {
        glGenVertexArrays(1, &o.gpu.vao);
        glGenBuffers(1, &o.gpu.vbo);
        glGenBuffers(1, &o.gpu.ibo);
    }
    glBindVertexArray(o.gpu.vao);
    glBindBuffer(GL_ARRAY_BUFFER, o.gpu.vbo);
    glBufferData(GL_ARRAY_BUFFER, m.v.size() * sizeof(Vertex), m.v.empty() ? nullptr : m.v.data(), GL_DYNAMIC_DRAW);
    setup_mesh_attribs();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, o.gpu.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m.idx.size() * sizeof(uint32_t), m.idx.empty() ? nullptr : m.idx.data(),
                 GL_DYNAMIC_DRAW);
    glBindVertexArray(0);
    o.gpu.nidx = static_cast<int>(m.idx.size());
    if (!m.v.empty()) o.center = m.center();
}

void upload_instances(Object &o) {
    glBindVertexArray(o.gpu.vao);
    glBindBuffer(GL_ARRAY_BUFFER, o.gpu.inst);
    glBufferData(GL_ARRAY_BUFFER, o.inst_data.size() * sizeof(float), o.inst_data.data(), GL_DYNAMIC_DRAW);
    glBindVertexArray(0);
    o.gpu.ninst = static_cast<int>(o.inst_data.size() / 11);
}

Object &new_object(Kind kind = STATIC) {
    int id = g_next_id++;
    Object &o = g_objects[id];
    o.id = id;
    o.kind = kind;
    return o;
}

int add_mesh(const MeshData &m, bool transparent) {
    Object &o = new_object();
    o.transparent = transparent;
    upload_mesh(o, m);
    return o.id;
}

int create_points(const float *xyz, const float *rgba, int n, float radius) {
    Object &o = new_object();
    o.instanced = true;
    glGenVertexArrays(1, &o.gpu.vao);
    glGenBuffers(1, &o.gpu.inst);
    glBindVertexArray(o.gpu.vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_sphere_vbo);
    setup_mesh_attribs();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_sphere_ibo);
    glBindBuffer(GL_ARRAY_BUFFER, o.gpu.inst);
    setup_instance_attribs();
    glBindVertexArray(0);
    o.gpu.nidx = g_sphere_nidx;
    o.inst_data.resize(static_cast<size_t>(n) * 11);
    Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
    bool translucent = false;
    for (int i = 0; i < n; ++i) {
        float *q = &o.inst_data[static_cast<size_t>(i) * 11];
        for (int c = 0; c < 3; ++c) q[c] = q[3 + c] = xyz ? xyz[i * 3 + c] : 0.0f;
        for (int c = 0; c < 4; ++c) q[6 + c] = rgba[i * 4 + c];
        q[10] = radius;
        translucent = translucent || rgba[i * 4 + 3] < 0.999f;
        lo = {std::min(lo.x, q[0]), std::min(lo.y, q[1]), std::min(lo.z, q[2])};
        hi = {std::max(hi.x, q[0]), std::max(hi.y, q[1]), std::max(hi.z, q[2])};
    }
    o.center = n ? (lo + hi) * 0.5f : Vec3();
    o.transparent = translucent;
    upload_instances(o);
    return o.id;
}

void delete_object(int id) {
    auto it = g_objects.find(id);
    if (it == g_objects.end()) return;
    Gpu &g = it->second.gpu;
    if (g.vao) glDeleteVertexArrays(1, &g.vao);
    if (g.vbo) glDeleteBuffers(1, &g.vbo);
    if (g.ibo) glDeleteBuffers(1, &g.ibo);
    if (g.inst) glDeleteBuffers(1, &g.inst);
    g_objects.erase(it);
}

// ---------------------------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------------------------

void camera_matrices() {
    float sp = std::sin(g_cam.phi), cp = std::cos(g_cam.phi);
    float st = std::sin(g_cam.theta), ct = std::cos(g_cam.theta);
    Vec3 dir(sp * ct, sp * st, cp);
    Vec3 up(-cp * ct, -cp * st, sp);  // tangent towards increasing elevation: never parallel to dir
    g_eye = g_cam.target + dir * g_cam.dist;
    Mat4 view = Mat4::look_at(g_eye, g_cam.target, up);
    float aspect = g_px_h > 0 ? static_cast<float>(g_px_w) / g_px_h : 1.0f;
    // the camera's field of view applies to the shorter side, so portrait canvases (phones) are not cropped
    float narrow = std::min(aspect, 1.0f);
    Mat4 proj;
    if (g_cam.ortho) {
        float h = g_cam.ortho_size / narrow, w = h * aspect;
        proj = Mat4::ortho(-w, w, -h, h, 0.1f, 200.0f);
    } else {
        float fovy = 2.0f * std::atan(std::tan(0.5f * g_cam.fov) / narrow);
        proj = Mat4::perspective(fovy, aspect, 0.1f, 200.0f);
    }
    g_view_proj = proj * view;
}

bool project(Vec3 p, float &sx, float &sy, float &depth) {
    Vec4 c = g_view_proj * Vec4{p.x, p.y, p.z, 1.0f};
    if (c.w <= 1e-6f) return false;
    float nx = c.x / c.w, ny = c.y / c.w;
    depth = c.z / c.w;
    sx = (nx * 0.5f + 0.5f) * g_css_w;
    sy = (1.0f - (ny * 0.5f + 0.5f)) * g_css_h;
    return depth > -1.0f && depth < 1.0f;
}

// ---------------------------------------------------------------------------------------------
// Animated objects
// ---------------------------------------------------------------------------------------------

void update_iso4d(Object &o) {
    if (std::fabs(o.time - o.last_time) < 1e-4f || o.nt == 0) return;
    o.last_time = o.time;
    float t = std::max(0.0f, std::min(o.time, static_cast<float>(o.nt - 1)));
    int k0 = static_cast<int>(std::floor(t)), k1 = std::min(k0 + 1, o.nt - 1);
    float s = t - k0;
    size_t cells = static_cast<size_t>(o.nx) * o.ny * o.nz;
    std::vector<float> f(cells);
    const float *A = &o.fields[k0 * cells], *B = &o.fields[k1 * cells];
    for (size_t i = 0; i < cells; ++i) f[i] = A[i] * (1 - s) + B[i] * s;
    upload_mesh(o, marching_tetrahedra(f.data(), o.nx, o.ny, o.nz, o.iso, o.lo, o.hi, o.color));
}

void update_tour(float dt) {
    if (!g_tour.active) return;
    g_tour.t += dt * g_tour.speed;
    const int d = g_tour.tour.d;
    g_tour.tour.frame_at(g_tour.t, g_tour.frame);
    const float sc = g_tour.scale;
    auto it = g_objects.find(g_tour.points);
    if (it != g_objects.end()) {
        Object &o = it->second;
        for (int i = 0; i < g_tour.n; ++i) {
            Vec3 p = project_row(g_tour.frame, d, &g_tour.X[static_cast<size_t>(i) * d]) * sc;
            float *q = &o.inst_data[static_cast<size_t>(i) * 11];
            q[0] = q[3] = p.x;
            q[1] = q[4] = p.y;
            q[2] = q[5] = p.z;
        }
        upload_instances(o);
    }
    if (g_tour.walk_m >= 2) {
        std::vector<Vec3> pts;
        for (int i = 0; i < g_tour.walk_m; ++i)
            pts.push_back(project_row(g_tour.frame, d, &g_tour.walk_rows[static_cast<size_t>(i) * d]) * sc);
        auto w = g_objects.find(g_tour.walk);
        if (w != g_objects.end()) upload_mesh(w->second, make_tube(pts, 0.035f, 8, g_tour.walk_color));
    }
    if (g_tour.arrow_k > 0) {
        Vec3 origin = project_row(g_tour.frame, d, g_tour.arrow_origin.data()) * sc;
        MeshData m;
        for (int k = 0; k < g_tour.arrow_k; ++k) {
            Vec3 v = project_row(g_tour.frame, d, &g_tour.arrow_vecs[static_cast<size_t>(k) * d]) * sc;
            m.append(make_arrow(origin, origin + v, 0.03f, g_tour.arrow_colors[k]));
        }
        auto a = g_objects.find(g_tour.arrows);
        if (a != g_objects.end()) upload_mesh(a->second, m);
    }
    if (g_tour.axes >= 0) {
        MeshData m;
        g_tour.axis_tips.assign(d, Vec3());
        std::vector<float> e(d, 0.0f);
        for (int j = 0; j < d; ++j) {
            std::fill(e.begin(), e.end(), 0.0f);
            e[j] = 1.0f;
            Vec3 tip = project_row(g_tour.frame, d, e.data()) * g_tour.axes_len;
            g_tour.axis_tips[j] = tip;
            if (length(tip) > 1e-3f) m.append(make_tube({Vec3(), tip}, 0.012f, 6, {0.75f, 0.75f, 0.78f, 0.55f}));
        }
        auto a = g_objects.find(g_tour.axes);
        if (a != g_objects.end()) upload_mesh(a->second, m);
    }
}

void set_prop(int target, int prop, float v) {
    if (target < 0) {
        switch (prop) {
            case THETA: g_cam.theta = v; break;
            case PHI: g_cam.phi = v; break;
            case DIST: g_cam.dist = v; break;
            case TARGET_Z: g_cam.target.z = v; break;
            case ORTHO_SIZE: g_cam.ortho_size = v; break;
            default: break;
        }
        return;
    }
    auto it = g_objects.find(target);
    if (it == g_objects.end()) return;
    Object &o = it->second;
    switch (prop) {
        case ALPHA: o.alpha = v; break;
        case REVEAL: o.reveal = v; break;
        case MORPH: o.morph = v; break;
        case SCALE: o.scale = v; break;
        case EMISSIVE: o.emissive = v; break;
        case TIME: o.time = v; break;
        case OFFSET_Z: o.offset_z = v; break;
        case VISIBLE: o.visible = v; break;
        default: break;
    }
}

float tween_value(const Tween &tw, float t) {
    float s = tw.dur > 0 ? (t - tw.start) / tw.dur : 1.0f;
    return tw.from + (tw.to - tw.from) * ease::apply(tw.easing, s);
}

// Object properties are a pure function of timeline time (so replaying from t = 0 restores
// the initial state): before a property's first tween it takes that tween's start value,
// afterwards the latest tween that has started decides. Camera tweens only act while they
// run (plus once at completion), leaving the camera free for the user and ambient rotation.
void apply_tweens() {
    std::map<std::pair<int, int>, std::vector<Tween *>> groups;
    for (Tween &tw : g_tweens) groups[{tw.target, tw.prop}].push_back(&tw);
    for (auto &kv : groups) {
        auto &list = kv.second;
        std::stable_sort(list.begin(), list.end(), [](const Tween *a, const Tween *b) { return a->start < b->start; });
        if (kv.first.first < 0) {
            for (Tween *tw : list) {
                if (g_time < tw->start) {
                    tw->done = false;
                } else if (g_time < tw->start + tw->dur) {
                    tw->done = false;
                    set_prop(tw->target, tw->prop, tween_value(*tw, g_time));
                } else if (!tw->done) {
                    tw->done = true;
                    set_prop(tw->target, tw->prop, tw->to);
                }
            }
            continue;
        }
        const Tween *active = nullptr;
        for (const Tween *tw : list)
            if (g_time >= tw->start) active = tw;
        float v = active ? tween_value(*active, g_time) : list.front()->from;
        set_prop(kv.first.first, kv.first.second, v);
    }
}

// ---------------------------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------------------------

void draw(const Object &o) {
    if (o.visible < 0.5f || o.alpha <= 0.001f || o.gpu.nidx == 0) return;
    if (o.instanced && o.gpu.ninst == 0) return;
    glUniform1f(uni("uMorph"), o.morph);
    glUniform1i(uni("uInstanced"), o.instanced ? 1 : 0);
    glUniform3f(uni("uOffset"), 0, 0, o.offset_z);
    glUniform1f(uni("uScale"), o.scale);
    glUniform3f(uni("uScaleOrigin"), o.scale_origin.x, o.scale_origin.y, o.scale_origin.z);
    glUniform1f(uni("uAlpha"), o.alpha);
    glUniform1f(uni("uReveal"), o.reveal);
    glUniform4f(uni("uTint"), o.tint.r, o.tint.g, o.tint.b, o.tint.a);
    glUniform1f(uni("uUnlit"), o.unlit ? 1.0f : 0.0f);
    glUniform1f(uni("uEmissive"), o.emissive);
    glUniform1i(uni("uClip"), (o.clip && g_clip_on) ? 1 : 0);
    glBindVertexArray(o.gpu.vao);
    if (o.instanced)
        glDrawElementsInstanced(GL_TRIANGLES, o.gpu.nidx, GL_UNSIGNED_INT, nullptr, o.gpu.ninst);
    else
        glDrawElements(GL_TRIANGLES, o.gpu.nidx, GL_UNSIGNED_INT, nullptr);
}

double g_frame_cost_ms = 0;  // moving average of the CPU time spent per frame (diagnostics)

void render(float dt);

void frame() {
    double now = emscripten_get_now();
    // clamp large gaps (tab switches) but stay close to real time even at a few frames per second
    float dt = g_last_ms > 0 ? static_cast<float>(std::min(0.5, (now - g_last_ms) / 1000.0)) : 0.0f;
    g_last_ms = now;
    if (g_paused) return;
    render(dt);
    double cost = emscripten_get_now() - now;
    g_frame_cost_ms = g_frame_cost_ms > 0 ? 0.9 * g_frame_cost_ms + 0.1 * cost : cost;
}

void render(float dt) {
    if (g_playing) g_time += dt * g_speed;
    apply_tweens();
    g_cam.theta += g_auto_rotate * dt;
    update_tour(dt);
    for (auto &kv : g_objects)
        if (kv.second.kind == ISO4D) update_iso4d(kv.second);

    camera_matrices();
    glViewport(0, 0, g_px_w, g_px_h);
    glClearColor(g_bg[0], g_bg[1], g_bg[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glUseProgram(g_prog);
    glUniformMatrix4fv(uni("uViewProj"), 1, GL_FALSE, g_view_proj.m);
    glUniform3f(uni("uEye"), g_eye.x, g_eye.y, g_eye.z);
    glUniform3f(uni("uClipLo"), g_clip_lo.x, g_clip_lo.y, g_clip_lo.z);
    glUniform3f(uni("uClipHi"), g_clip_hi.x, g_clip_hi.y, g_clip_hi.z);

    std::vector<const Object *> opaque, transparent;
    for (auto &kv : g_objects) {
        const Object &o = kv.second;
        bool see_through = o.transparent || o.alpha < 0.999f || o.tint.a < 0.999f;
        (see_through ? transparent : opaque).push_back(&o);
    }
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    for (const Object *o : opaque) draw(*o);
    std::sort(transparent.begin(), transparent.end(), [](const Object *a, const Object *b) {
        return length(a->center - g_eye) > length(b->center - g_eye);
    });
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    for (const Object *o : transparent) draw(*o);
    glDepthMask(GL_TRUE);
    glBindVertexArray(0);
}

RGBA rgba(float r, float g, float b, float a) { return {r, g, b, a}; }

}  // namespace

// =============================================================================================
// C API (called from TypeScript through cwrap)
// =============================================================================================
extern "C" {

EMSCRIPTEN_KEEPALIVE int viz_init(const char *selector) {
    g_selector = selector;
    EmscriptenWebGLContextAttributes attr;
    emscripten_webgl_init_context_attributes(&attr);
    attr.majorVersion = 2;
    attr.minorVersion = 0;
    attr.antialias = true;
    attr.alpha = false;
    attr.depth = true;
    attr.premultipliedAlpha = false;
    attr.preserveDrawingBuffer = false;
    g_ctx = emscripten_webgl_create_context(selector, &attr);
    if (g_ctx <= 0) return 0;
    emscripten_webgl_make_context_current(g_ctx);
    g_prog = glCreateProgram();
    glAttachShader(g_prog, compile(GL_VERTEX_SHADER, kVS));
    glAttachShader(g_prog, compile(GL_FRAGMENT_SHADER, kFS));
    glLinkProgram(g_prog);
    GLint ok = 0;
    glGetProgramiv(g_prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(g_prog, sizeof(log), nullptr, log);
        std::printf("link error: %s\n", log);
        return 0;
    }
    MeshData sphere = make_icosphere(2);
    glGenBuffers(1, &g_sphere_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_sphere_vbo);
    glBufferData(GL_ARRAY_BUFFER, sphere.v.size() * sizeof(Vertex), sphere.v.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &g_sphere_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_sphere_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere.idx.size() * sizeof(uint32_t), sphere.idx.data(), GL_STATIC_DRAW);
    g_sphere_nidx = static_cast<int>(sphere.idx.size());
    emscripten_set_main_loop(frame, 0, false);
    return 1;
}

EMSCRIPTEN_KEEPALIVE void viz_resize(int px_w, int px_h, float css_w, float css_h) {
    g_px_w = std::max(px_w, 1);
    g_px_h = std::max(px_h, 1);
    g_css_w = css_w;
    g_css_h = css_h;
    emscripten_set_canvas_element_size(g_selector.c_str(), g_px_w, g_px_h);
}

EMSCRIPTEN_KEEPALIVE void viz_clear() {
    std::vector<int> ids;
    for (auto &kv : g_objects) ids.push_back(kv.first);
    for (int id : ids) delete_object(id);
    g_tweens.clear();
    g_tour = TourState();
    g_time = 0;
}

EMSCRIPTEN_KEEPALIVE void viz_remove(int id) { delete_object(id); }
EMSCRIPTEN_KEEPALIVE void viz_set_paused(int paused) {
    g_paused = paused != 0;
    g_last_ms = 0;
}
EMSCRIPTEN_KEEPALIVE void viz_set_background(float r, float g, float b) {
    g_bg[0] = r;
    g_bg[1] = g;
    g_bg[2] = b;
}
EMSCRIPTEN_KEEPALIVE void viz_set_clip(int on, float x0, float x1, float y0, float y1, float z0, float z1) {
    g_clip_on = on != 0;
    g_clip_lo = {x0, y0, z0};
    g_clip_hi = {x1, y1, z1};
}

// --- objects ---------------------------------------------------------------------------------

EMSCRIPTEN_KEEPALIVE int viz_add_surface(const float *hA, const float *hB, int nu, int nv, float u0, float u1,
                                         float v0, float v1, float zscale, float r, float g, float b, float a,
                                         int checker) {
    int id = add_mesh(make_surface(hA, hB, nu, nv, u0, u1, v0, v1, zscale, rgba(r, g, b, a), checker != 0), a < 0.999f);
    g_objects[id].clip = true;
    return id;
}

EMSCRIPTEN_KEEPALIVE int viz_add_isosurface(const float *field, int nx, int ny, int nz, float iso, float x0, float x1,
                                            float y0, float y1, float z0, float z1, float r, float g, float b,
                                            float a) {
    return add_mesh(marching_tetrahedra(field, nx, ny, nz, iso, {x0, y0, z0}, {x1, y1, z1}, rgba(r, g, b, a)),
                    a < 0.999f);
}

// A field sampled at nt "times" (a 4th coordinate); tween TIME to sweep through it.
EMSCRIPTEN_KEEPALIVE int viz_add_isosurface_4d(const float *fields, int nt, int nx, int ny, int nz, float iso,
                                               float x0, float x1, float y0, float y1, float z0, float z1, float r,
                                               float g, float b, float a) {
    Object &o = new_object(ISO4D);
    o.fields.assign(fields, fields + static_cast<size_t>(nt) * nx * ny * nz);
    o.nt = nt;
    o.nx = nx;
    o.ny = ny;
    o.nz = nz;
    o.iso = iso;
    o.lo = {x0, y0, z0};
    o.hi = {x1, y1, z1};
    o.color = rgba(r, g, b, a);
    o.transparent = a < 0.999f;
    o.time = 0;
    update_iso4d(o);
    return o.id;
}

EMSCRIPTEN_KEEPALIVE int viz_add_isolines(const float *field, int nx, int ny, float iso, float x0, float x1, float y0,
                                          float y1, float z, float radius, float r, float g, float b, float a) {
    return add_mesh(make_segments(marching_squares(field, nx, ny, iso, x0, x1, y0, y1, z), radius, rgba(r, g, b, a)),
                    a < 0.999f);
}

EMSCRIPTEN_KEEPALIVE int viz_add_points(const float *xyz, const float *colors, int n, float radius, int pickable) {
    int id = create_points(xyz, colors, n, radius);
    g_objects[id].pickable = pickable != 0;
    return id;
}

// Morph target for a point cloud (manim Transform between two layouts).
EMSCRIPTEN_KEEPALIVE void viz_points_target(int id, const float *xyz) {
    auto it = g_objects.find(id);
    if (it == g_objects.end() || !it->second.instanced) return;
    Object &o = it->second;
    for (int i = 0; i < o.gpu.ninst; ++i)
        for (int c = 0; c < 3; ++c) o.inst_data[static_cast<size_t>(i) * 11 + 3 + c] = xyz[i * 3 + c];
    upload_instances(o);
}

EMSCRIPTEN_KEEPALIVE int viz_add_tube(const float *xyz, int n, float radius, float r, float g, float b, float a) {
    std::vector<Vec3> pts;
    for (int i = 0; i < n; ++i) pts.push_back({xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]});
    return add_mesh(make_tube(pts, radius, 10, rgba(r, g, b, a)), a < 0.999f);
}

EMSCRIPTEN_KEEPALIVE int viz_add_arrow(float x0, float y0, float z0, float x1, float y1, float z1, float radius,
                                       float r, float g, float b, float a) {
    return add_mesh(make_arrow({x0, y0, z0}, {x1, y1, z1}, radius, rgba(r, g, b, a)), a < 0.999f);
}

EMSCRIPTEN_KEEPALIVE int viz_add_plane_z(float z, float x0, float x1, float y0, float y1, float r, float g, float b,
                                         float a) {
    return add_mesh(make_quad_z(z, x0, x1, y0, y1, rgba(r, g, b, a)), a < 0.999f);
}

EMSCRIPTEN_KEEPALIVE int viz_add_box(float x0, float x1, float y0, float y1, float z0, float z1, float radius, float r,
                                     float g, float b, float a) {
    return add_mesh(make_box_edges({x0, y0, z0}, {x1, y1, z1}, radius, rgba(r, g, b, a)), a < 0.999f);
}

EMSCRIPTEN_KEEPALIVE int viz_add_floor_grid(float x0, float x1, float y0, float y1, float z, float step, float radius,
                                            float r, float g, float b, float a) {
    return add_mesh(make_floor_grid(x0, x1, y0, y1, z, step, radius, rgba(r, g, b, a)), a < 0.999f);
}

EMSCRIPTEN_KEEPALIVE void viz_set(int id, int prop, float value) { set_prop(id, prop, value); }

EMSCRIPTEN_KEEPALIVE void viz_set_unlit(int id, int unlit) {
    auto it = g_objects.find(id);
    if (it != g_objects.end()) it->second.unlit = unlit != 0;
}

EMSCRIPTEN_KEEPALIVE void viz_set_scale_origin(int id, float x, float y, float z) {
    auto it = g_objects.find(id);
    if (it != g_objects.end()) it->second.scale_origin = {x, y, z};
}

EMSCRIPTEN_KEEPALIVE void viz_set_clip_object(int id, int on) {
    auto it = g_objects.find(id);
    if (it != g_objects.end()) it->second.clip = on != 0;
}

// --- timeline --------------------------------------------------------------------------------

EMSCRIPTEN_KEEPALIVE void viz_tween(int target, int prop, float from, float to, float start, float dur, int easing) {
    g_tweens.push_back({target, prop, from, to, start, dur, easing});
}
EMSCRIPTEN_KEEPALIVE void viz_clear_tweens() { g_tweens.clear(); }
EMSCRIPTEN_KEEPALIVE void viz_set_time(float t) { g_time = t; }
EMSCRIPTEN_KEEPALIVE float viz_time() { return g_time; }
EMSCRIPTEN_KEEPALIVE void viz_play(int playing, float speed) {
    g_playing = playing != 0;
    g_speed = speed;
}

// --- camera ----------------------------------------------------------------------------------

EMSCRIPTEN_KEEPALIVE void viz_camera(float theta, float phi, float dist, float tx, float ty, float tz) {
    g_cam.theta = theta;
    g_cam.phi = phi;
    g_cam.dist = dist;
    g_cam.target = {tx, ty, tz};
}
EMSCRIPTEN_KEEPALIVE void viz_camera_ortho(int on, float size) {
    g_cam.ortho = on != 0;
    g_cam.ortho_size = size;
}
EMSCRIPTEN_KEEPALIVE void viz_auto_rotate(float speed) { g_auto_rotate = speed; }
// User interaction: orbit / zoom, cancelling scripted camera moves.
EMSCRIPTEN_KEEPALIVE void viz_orbit(float dtheta, float dphi) {
    g_tweens.erase(std::remove_if(g_tweens.begin(), g_tweens.end(), [](const Tween &t) { return t.target < 0; }),
                   g_tweens.end());
    g_cam.theta += dtheta;
    g_cam.phi = std::min(std::max(g_cam.phi + dphi, 0.02f), 3.12f);
}
EMSCRIPTEN_KEEPALIVE void viz_zoom(float factor) {
    g_tweens.erase(std::remove_if(g_tweens.begin(), g_tweens.end(),
                                  [](const Tween &t) { return t.target < 0 && (t.prop == DIST || t.prop == ORTHO_SIZE); }),
                   g_tweens.end());
    g_cam.dist = std::min(std::max(g_cam.dist * factor, 2.0f), 60.0f);
    g_cam.ortho_size = std::min(std::max(g_cam.ortho_size * factor, 0.5f), 30.0f);
}
EMSCRIPTEN_KEEPALIVE float viz_camera_theta() { return g_cam.theta; }
EMSCRIPTEN_KEEPALIVE float viz_frame_cost_ms() { return static_cast<float>(g_frame_cost_ms); }

// Screen position (CSS px) of a world point for HTML labels: out = [x, y, visible].
EMSCRIPTEN_KEEPALIVE void viz_project(float x, float y, float z, float *out) {
    camera_matrices();
    float sx = 0, sy = 0, depth = 0;
    bool vis = project({x, y, z}, sx, sy, depth);
    out[0] = sx;
    out[1] = sy;
    out[2] = vis ? 1.0f : 0.0f;
}

// Nearest pickable point within max_px of a CSS-pixel position: returns (object id << 20) | index, or -1.
EMSCRIPTEN_KEEPALIVE int viz_pick(float css_x, float css_y, float max_px) {
    camera_matrices();
    int best = -1;
    float best_d = max_px * max_px;
    for (auto &kv : g_objects) {
        const Object &o = kv.second;
        if (!o.pickable || !o.instanced || o.visible < 0.5f || o.alpha < 0.05f) continue;
        for (int i = 0; i < o.gpu.ninst; ++i) {
            const float *q = &o.inst_data[static_cast<size_t>(i) * 11];
            Vec3 p = lerp(Vec3(q[0], q[1], q[2]), Vec3(q[3], q[4], q[5]), o.morph);
            float sx, sy, depth;
            if (!project(p, sx, sy, depth)) continue;
            float d2 = (sx - css_x) * (sx - css_x) + (sy - css_y) * (sy - css_y);
            if (d2 < best_d) {
                best_d = d2;
                best = (o.id << 20) | i;
            }
        }
    }
    return best;
}

// --- grand tour --------------------------------------------------------------------------------

EMSCRIPTEN_KEEPALIVE int viz_tour_begin(const float *X, int n, int d, const float *colors, float radius,
                                        const float *keyframes, int nk, float scale) {
    g_tour = TourState();
    g_tour.active = true;
    g_tour.n = n;
    g_tour.tour.d = d;
    g_tour.X.assign(X, X + static_cast<size_t>(n) * d);
    g_tour.tour.keyframes.assign(keyframes, keyframes + static_cast<size_t>(nk) * 3 * d);
    g_tour.scale = scale;
    g_tour.points = create_points(nullptr, colors, n, radius);
    g_objects[g_tour.points].pickable = true;
    g_objects[g_tour.points].kind = TOUR_POINTS;
    update_tour(0);
    return g_tour.points;
}

EMSCRIPTEN_KEEPALIVE int viz_tour_walk(const float *rows, int m, float r, float g, float b, float a) {
    if (!g_tour.active) return -1;
    g_tour.walk_rows.assign(rows, rows + static_cast<size_t>(m) * g_tour.tour.d);
    g_tour.walk_m = m;
    g_tour.walk_color = rgba(r, g, b, a);
    g_tour.walk = new_object(TOUR_WALK).id;
    update_tour(0);
    return g_tour.walk;
}

EMSCRIPTEN_KEEPALIVE int viz_tour_arrows(const float *origin, const float *vecs, int k, const float *colors) {
    if (!g_tour.active) return -1;
    const int d = g_tour.tour.d;
    g_tour.arrow_origin.assign(origin, origin + d);
    g_tour.arrow_vecs.assign(vecs, vecs + static_cast<size_t>(k) * d);
    g_tour.arrow_k = k;
    g_tour.arrow_colors.clear();
    for (int i = 0; i < k; ++i)
        g_tour.arrow_colors.push_back(rgba(colors[i * 4], colors[i * 4 + 1], colors[i * 4 + 2], colors[i * 4 + 3]));
    g_tour.arrows = new_object(TOUR_ARROWS).id;
    update_tour(0);
    return g_tour.arrows;
}

EMSCRIPTEN_KEEPALIVE int viz_tour_axes(float len) {
    if (!g_tour.active) return -1;
    g_tour.axes_len = len;
    Object &o = new_object(TOUR_AXES);
    o.transparent = true;
    g_tour.axes = o.id;
    update_tour(0);
    return g_tour.axes;
}

EMSCRIPTEN_KEEPALIVE void viz_tour_speed(float speed) { g_tour.speed = speed; }
EMSCRIPTEN_KEEPALIVE void viz_tour_time(float t) {
    g_tour.t = t;
    update_tour(0);
}
EMSCRIPTEN_KEEPALIVE float viz_tour_get_time() { return g_tour.t; }

EMSCRIPTEN_KEEPALIVE void viz_tour_axis_tip(int j, float *out) {
    Vec3 p = (j >= 0 && j < static_cast<int>(g_tour.axis_tips.size())) ? g_tour.axis_tips[j] : Vec3();
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
}

}  // extern "C"
