// CPU geometry for the renderer: surfaces with morph targets, tubes, cones, spheres,
// iso-surfaces (marching tetrahedra), iso-lines (marching squares) and the
// grand-tour projection. Pure C++17 - no OpenGL here, so it is unit-tested natively.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "vmath.hpp"

// One vertex: position/normal in state A, position/normal in state B (the morph
// target, used for manim-style Transform), RGBA colour and a reveal parameter in
// [0, 1] (used for manim-style Create: fragments with param > reveal are dropped).
struct Vertex {
    float pa[3], na[3], pb[3], nb[3], col[4], param;
};
static_assert(sizeof(Vertex) == 17 * sizeof(float), "Vertex must be tightly packed");

struct MeshData {
    std::vector<Vertex> v;
    std::vector<uint32_t> idx;
    void append(const MeshData &other);
    Vec3 center() const;  // centre of the bounding box (used to sort transparent objects)
};

struct RGBA {
    float r, g, b, a;
};

Vertex make_vertex(Vec3 p, Vec3 n, RGBA c, float param);

// z = h(u, v) * zscale on a regular grid, h indexed h[j * nu + i] (i along x, j along y).
// hB (may be null) is the morph target with the same topology. NaN heights cut holes.
// checker: per-cell vertices with alternating shades (manim's checkerboard surfaces).
MeshData make_surface(const float *hA, const float *hB, int nu, int nv, float u0, float u1, float v0, float v1,
                      float zscale, RGBA color, bool checker);

// Tube of radius r around a polyline; param = arc-length fraction (for "Create").
MeshData make_tube(const std::vector<Vec3> &pts, float radius, int sides, RGBA color);
MeshData make_cone(Vec3 base, Vec3 tip, float radius, int sides, RGBA color);
MeshData make_arrow(Vec3 from, Vec3 to, float radius, RGBA color);
MeshData make_icosphere(int subdivisions);
MeshData make_quad_z(float z, float x0, float x1, float y0, float y1, RGBA color);
MeshData make_segments(const std::vector<std::pair<Vec3, Vec3>> &segs, float radius, RGBA color);
MeshData make_box_edges(Vec3 lo, Vec3 hi, float radius, RGBA color);
MeshData make_floor_grid(float x0, float x1, float y0, float y1, float z, float step, float radius, RGBA color);

// Iso-surface {field = iso} of a scalar field sampled on an nx*ny*nz grid spanning
// [lo, hi] (index (k * ny + j) * nx + i). Normals follow the field gradient.
MeshData marching_tetrahedra(const float *field, int nx, int ny, int nz, float iso, Vec3 lo, Vec3 hi, RGBA color);

// Iso-line segments {field = iso} of an nx*ny field on [x0,x1]x[y0,y1], placed at height z.
std::vector<std::pair<Vec3, Vec3>> marching_squares(const float *field, int nx, int ny, float iso, float x0, float x1,
                                                   float y0, float y1, float z);

// Grand tour: interpolate between d x 3 orthonormal keyframes (column-major,
// frame[k * d * 3 + c * d + j]) and project d-dimensional rows onto the current frame.
struct Tour {
    int d = 0;
    std::vector<float> keyframes;  // nk * 3 * d
    int count() const { return d ? static_cast<int>(keyframes.size()) / (3 * d) : 0; }
    // Frame at time t (in keyframe units, cyclic), smooth easing between keyframes;
    // writes 3 orthonormal columns of length d into out (out[c * d + j]).
    void frame_at(float t, std::vector<float> &out) const;
};

void gram_schmidt(std::vector<float> &cols, int d, int ncols);
Vec3 project_row(const std::vector<float> &frame, int d, const float *x);
