#include "geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>

namespace {
constexpr float kPi = 3.14159265358979f;

void put3(float *dst, Vec3 v) {
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
}
}  // namespace

Vertex make_vertex(Vec3 p, Vec3 n, RGBA c, float param) {
    Vertex v{};
    put3(v.pa, p);
    put3(v.na, n);
    put3(v.pb, p);
    put3(v.nb, n);
    v.col[0] = c.r;
    v.col[1] = c.g;
    v.col[2] = c.b;
    v.col[3] = c.a;
    v.param = param;
    return v;
}

void MeshData::append(const MeshData &other) {
    auto base = static_cast<uint32_t>(v.size());
    v.insert(v.end(), other.v.begin(), other.v.end());
    idx.reserve(idx.size() + other.idx.size());
    for (uint32_t i : other.idx) idx.push_back(base + i);
}

Vec3 MeshData::center() const {
    if (v.empty()) return {};
    Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
    for (const Vertex &q : v) {
        lo = {std::min(lo.x, q.pa[0]), std::min(lo.y, q.pa[1]), std::min(lo.z, q.pa[2])};
        hi = {std::max(hi.x, q.pa[0]), std::max(hi.y, q.pa[1]), std::max(hi.z, q.pa[2])};
    }
    return (lo + hi) * 0.5f;
}

// ---------------------------------------------------------------------------------------------
// Height-field surfaces
// ---------------------------------------------------------------------------------------------

static Vec3 grid_normal(const float *h, int nu, int nv, int i, int j, float du, float dv, float zscale) {
    int i0 = std::max(i - 1, 0), i1 = std::min(i + 1, nu - 1);
    int j0 = std::max(j - 1, 0), j1 = std::min(j + 1, nv - 1);
    float dzdx = (h[j * nu + i1] - h[j * nu + i0]) * zscale / ((i1 - i0) * du);
    float dzdy = (h[j1 * nu + i] - h[j0 * nu + i]) * zscale / ((j1 - j0) * dv);
    if (!std::isfinite(dzdx)) dzdx = 0;
    if (!std::isfinite(dzdy)) dzdy = 0;
    return normalize(Vec3(-dzdx, -dzdy, 1.0f));
}

MeshData make_surface(const float *hA, const float *hB, int nu, int nv, float u0, float u1, float v0, float v1,
                      float zscale, RGBA color, bool checker) {
    MeshData m;
    if (nu < 2 || nv < 2) return m;
    if (!hB) hB = hA;
    float du = (u1 - u0) / (nu - 1), dv = (v1 - v0) / (nv - 1);
    auto make = [&](int i, int j, RGBA c) {
        float x = u0 + du * i, y = v0 + dv * j;
        float xn = (2.0f * i) / (nu - 1) - 1.0f, yn = (2.0f * j) / (nv - 1) - 1.0f;
        float param = std::sqrt(xn * xn + yn * yn) / std::sqrt(2.0f);
        Vertex q = make_vertex(Vec3(x, y, hA[j * nu + i] * zscale), grid_normal(hA, nu, nv, i, j, du, dv, zscale), c,
                               param);
        put3(q.pb, Vec3(x, y, hB[j * nu + i] * zscale));
        put3(q.nb, grid_normal(hB, nu, nv, i, j, du, dv, zscale));
        return q;
    };
    auto finite = [&](int i, int j) { return std::isfinite(hA[j * nu + i]) && std::isfinite(hB[j * nu + i]); };
    if (!checker) {
        for (int j = 0; j < nv; ++j)
            for (int i = 0; i < nu; ++i) m.v.push_back(make(i, j, color));
        for (int j = 0; j + 1 < nv; ++j)
            for (int i = 0; i + 1 < nu; ++i) {
                if (!finite(i, j) || !finite(i + 1, j) || !finite(i, j + 1) || !finite(i + 1, j + 1)) continue;
                uint32_t a = j * nu + i, b = a + 1, c = a + nu + 1, d = a + nu;
                m.idx.insert(m.idx.end(), {a, b, c, a, c, d});
            }
        return m;
    }
    RGBA dark{color.r * 0.78f, color.g * 0.78f, color.b * 0.78f, color.a};
    for (int j = 0; j + 1 < nv; ++j)
        for (int i = 0; i + 1 < nu; ++i) {
            if (!finite(i, j) || !finite(i + 1, j) || !finite(i, j + 1) || !finite(i + 1, j + 1)) continue;
            RGBA c = ((i / 2 + j / 2) % 2 == 0) ? color : dark;
            auto base = static_cast<uint32_t>(m.v.size());
            m.v.push_back(make(i, j, c));
            m.v.push_back(make(i + 1, j, c));
            m.v.push_back(make(i + 1, j + 1, c));
            m.v.push_back(make(i, j + 1, c));
            m.idx.insert(m.idx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        }
    return m;
}

// ---------------------------------------------------------------------------------------------
// Tubes, cones, arrows, spheres, quads
// ---------------------------------------------------------------------------------------------

MeshData make_tube(const std::vector<Vec3> &input, float radius, int sides, RGBA color) {
    MeshData m;
    std::vector<Vec3> pts;
    for (const Vec3 &p : input)
        if (pts.empty() || length(p - pts.back()) > 1e-6f) pts.push_back(p);
    if (pts.size() < 2) return m;
    size_t n = pts.size();
    std::vector<float> arc(n, 0.0f);
    for (size_t i = 1; i < n; ++i) arc[i] = arc[i - 1] + length(pts[i] - pts[i - 1]);
    float total = arc.back() > 0 ? arc.back() : 1.0f;
    Vec3 t0 = normalize(pts[1] - pts[0]);
    Vec3 ref = std::fabs(t0.z) < 0.9f ? Vec3(0, 0, 1) : Vec3(1, 0, 0);
    Vec3 normal = normalize(cross(t0, ref));
    for (size_t i = 0; i < n; ++i) {
        Vec3 t = normalize(pts[std::min(i + 1, n - 1)] - pts[i > 0 ? i - 1 : 0]);
        normal = normalize(normal - t * dot(normal, t));  // parallel transport by projection
        Vec3 bin = cross(t, normal);
        for (int s = 0; s < sides; ++s) {
            float a = 2.0f * kPi * s / sides;
            Vec3 dir = normal * std::cos(a) + bin * std::sin(a);
            m.v.push_back(make_vertex(pts[i] + dir * radius, dir, color, arc[i] / total));
        }
    }
    for (size_t i = 0; i + 1 < n; ++i)
        for (int s = 0; s < sides; ++s) {
            uint32_t a = i * sides + s, b = i * sides + (s + 1) % sides;
            uint32_t c = (i + 1) * sides + (s + 1) % sides, d = (i + 1) * sides + s;
            m.idx.insert(m.idx.end(), {a, b, c, a, c, d});
        }
    return m;
}

MeshData make_cone(Vec3 base, Vec3 tip, float radius, int sides, RGBA color) {
    MeshData m;
    Vec3 axis = tip - base;
    float h = length(axis);
    if (h < 1e-6f) return m;
    Vec3 t = axis * (1.0f / h);
    Vec3 ref = std::fabs(t.z) < 0.9f ? Vec3(0, 0, 1) : Vec3(1, 0, 0);
    Vec3 n1 = normalize(cross(t, ref)), n2 = cross(t, n1);
    for (int s = 0; s < sides; ++s) {
        float a0 = 2.0f * kPi * s / sides, a1 = 2.0f * kPi * (s + 1) / sides;
        Vec3 d0 = n1 * std::cos(a0) + n2 * std::sin(a0), d1 = n1 * std::cos(a1) + n2 * std::sin(a1);
        Vec3 nn0 = normalize(d0 * h + t * radius), nn1 = normalize(d1 * h + t * radius);
        auto b = static_cast<uint32_t>(m.v.size());
        m.v.push_back(make_vertex(base + d0 * radius, nn0, color, 1.0f));
        m.v.push_back(make_vertex(base + d1 * radius, nn1, color, 1.0f));
        m.v.push_back(make_vertex(tip, normalize(nn0 + nn1), color, 1.0f));
        m.v.push_back(make_vertex(base, t * -1.0f, color, 1.0f));
        m.v.push_back(make_vertex(base + d1 * radius, t * -1.0f, color, 1.0f));
        m.v.push_back(make_vertex(base + d0 * radius, t * -1.0f, color, 1.0f));
        m.idx.insert(m.idx.end(), {b, b + 1, b + 2, b + 3, b + 4, b + 5});
    }
    return m;
}

MeshData make_arrow(Vec3 from, Vec3 to, float radius, RGBA color) {
    Vec3 d = to - from;
    float len = length(d);
    MeshData m;
    if (len < 1e-5f) return m;
    float head = std::min(len * 0.35f, radius * 7.0f);
    Vec3 neck = to - d * (head / len);
    m = make_tube({from, neck}, radius, 10, color);
    m.append(make_cone(neck, to, radius * 2.6f, 16, color));
    return m;
}

MeshData make_icosphere(int subdivisions) {
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
    std::vector<Vec3> p = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                           {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    for (Vec3 &q : p) q = normalize(q);
    std::vector<std::array<uint32_t, 3>> f = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
                                              {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                                              {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
                                              {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    for (int s = 0; s < subdivisions; ++s) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> mid;
        auto midpoint = [&](uint32_t a, uint32_t b) {
            auto key = std::make_pair(std::min(a, b), std::max(a, b));
            auto it = mid.find(key);
            if (it != mid.end()) return it->second;
            p.push_back(normalize(p[a] + p[b]));
            auto id = static_cast<uint32_t>(p.size() - 1);
            mid[key] = id;
            return id;
        };
        std::vector<std::array<uint32_t, 3>> next;
        for (auto &tri : f) {
            uint32_t a = midpoint(tri[0], tri[1]), b = midpoint(tri[1], tri[2]), c = midpoint(tri[2], tri[0]);
            next.push_back({tri[0], a, c});
            next.push_back({tri[1], b, a});
            next.push_back({tri[2], c, b});
            next.push_back({a, b, c});
        }
        f.swap(next);
    }
    MeshData m;
    for (const Vec3 &q : p) m.v.push_back(make_vertex(q, q, {1, 1, 1, 1}, 0.0f));
    for (auto &tri : f) m.idx.insert(m.idx.end(), {tri[0], tri[1], tri[2]});
    return m;
}

MeshData make_quad_z(float z, float x0, float x1, float y0, float y1, RGBA color) {
    MeshData m;
    Vec3 n(0, 0, 1);
    m.v.push_back(make_vertex({x0, y0, z}, n, color, 1.0f));
    m.v.push_back(make_vertex({x1, y0, z}, n, color, 1.0f));
    m.v.push_back(make_vertex({x1, y1, z}, n, color, 1.0f));
    m.v.push_back(make_vertex({x0, y1, z}, n, color, 1.0f));
    m.idx = {0, 1, 2, 0, 2, 3};
    return m;
}

MeshData make_segments(const std::vector<std::pair<Vec3, Vec3>> &segs, float radius, RGBA color) {
    MeshData m;
    for (const auto &s : segs) m.append(make_tube({s.first, s.second}, radius, 6, color));
    return m;
}

MeshData make_box_edges(Vec3 lo, Vec3 hi, float radius, RGBA color) {
    std::vector<std::pair<Vec3, Vec3>> e;
    Vec3 c[8];
    for (int i = 0; i < 8; ++i) c[i] = {(i & 1) ? hi.x : lo.x, (i & 2) ? hi.y : lo.y, (i & 4) ? hi.z : lo.z};
    int pairs[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    for (auto &p : pairs) e.push_back({c[p[0]], c[p[1]]});
    return make_segments(e, radius, color);
}

MeshData make_floor_grid(float x0, float x1, float y0, float y1, float z, float step, float radius, RGBA color) {
    std::vector<std::pair<Vec3, Vec3>> e;
    for (float x = std::ceil(x0 / step) * step; x <= x1 + 1e-4f; x += step) e.push_back({{x, y0, z}, {x, y1, z}});
    for (float y = std::ceil(y0 / step) * step; y <= y1 + 1e-4f; y += step) e.push_back({{x0, y, z}, {x1, y, z}});
    return make_segments(e, radius, color);
}

// ---------------------------------------------------------------------------------------------
// Marching tetrahedra (6 tetrahedra per cube around the 0-6 diagonal; consistent across cubes)
// ---------------------------------------------------------------------------------------------

MeshData marching_tetrahedra(const float *field, int nx, int ny, int nz, float iso, Vec3 lo, Vec3 hi, RGBA color) {
    MeshData m;
    if (nx < 2 || ny < 2 || nz < 2) return m;
    auto at = [&](int i, int j, int k) { return field[(static_cast<size_t>(k) * ny + j) * nx + i]; };
    Vec3 step((hi.x - lo.x) / (nx - 1), (hi.y - lo.y) / (ny - 1), (hi.z - lo.z) / (nz - 1));
    auto grad = [&](int i, int j, int k) {
        int i0 = std::max(i - 1, 0), i1 = std::min(i + 1, nx - 1);
        int j0 = std::max(j - 1, 0), j1 = std::min(j + 1, ny - 1);
        int k0 = std::max(k - 1, 0), k1 = std::min(k + 1, nz - 1);
        return Vec3((at(i1, j, k) - at(i0, j, k)) / ((i1 - i0) * step.x),
                    (at(i, j1, k) - at(i, j0, k)) / ((j1 - j0) * step.y),
                    (at(i, j, k1) - at(i, j, k0)) / ((k1 - k0) * step.z));
    };
    static const int corner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                     {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    static const int tets[6][4] = {{0, 5, 1, 6}, {0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}};
    float span = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
    for (int k = 0; k + 1 < nz; ++k)
        for (int j = 0; j + 1 < ny; ++j)
            for (int i = 0; i + 1 < nx; ++i) {
                float val[8];
                Vec3 pos[8], gr[8];
                bool any_in = false, any_out = false;
                for (int c = 0; c < 8; ++c) {
                    int ci = i + corner[c][0], cj = j + corner[c][1], ck = k + corner[c][2];
                    val[c] = at(ci, cj, ck);
                    (val[c] >= iso ? any_in : any_out) = true;
                }
                if (!any_in || !any_out) continue;
                for (int c = 0; c < 8; ++c) {
                    int ci = i + corner[c][0], cj = j + corner[c][1], ck = k + corner[c][2];
                    pos[c] = {lo.x + step.x * ci, lo.y + step.y * cj, lo.z + step.z * ck};
                    gr[c] = grad(ci, cj, ck);
                }
                for (const auto &tet : tets) {
                    int in[4], out[4], ni = 0, no = 0;
                    for (int q = 0; q < 4; ++q) (val[tet[q]] >= iso ? in[ni++] : out[no++]) = tet[q];
                    if (ni == 0 || ni == 4) continue;
                    auto edge = [&](int a, int b) {
                        float denom = val[b] - val[a];
                        float t = std::fabs(denom) > 1e-12f ? (iso - val[a]) / denom : 0.5f;
                        Vec3 p = lerp(pos[a], pos[b], t);
                        Vec3 n = normalize(lerp(gr[a], gr[b], t));
                        float param = length(p - (lo + hi) * 0.5f) / (0.5f * span * std::sqrt(3.0f));
                        return make_vertex(p, n, color, std::min(param, 1.0f));
                    };
                    auto base = static_cast<uint32_t>(m.v.size());
                    if (ni == 1 || ni == 3) {
                        int apex = ni == 1 ? in[0] : out[0];
                        const int *others = ni == 1 ? out : in;
                        m.v.push_back(edge(apex, others[0]));
                        m.v.push_back(edge(apex, others[1]));
                        m.v.push_back(edge(apex, others[2]));
                        m.idx.insert(m.idx.end(), {base, base + 1, base + 2});
                    } else {
                        m.v.push_back(edge(in[0], out[0]));
                        m.v.push_back(edge(in[0], out[1]));
                        m.v.push_back(edge(in[1], out[1]));
                        m.v.push_back(edge(in[1], out[0]));
                        m.idx.insert(m.idx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
                    }
                }
            }
    return m;
}

std::vector<std::pair<Vec3, Vec3>> marching_squares(const float *field, int nx, int ny, float iso, float x0, float x1,
                                                   float y0, float y1, float z) {
    std::vector<std::pair<Vec3, Vec3>> segs;
    float dx = (x1 - x0) / (nx - 1), dy = (y1 - y0) / (ny - 1);
    auto at = [&](int i, int j) { return field[j * nx + i]; };
    for (int j = 0; j + 1 < ny; ++j)
        for (int i = 0; i + 1 < nx; ++i) {
            float a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
            int idx = (a > iso ? 1 : 0) | (b > iso ? 2 : 0) | (c > iso ? 4 : 0) | (d > iso ? 8 : 0);
            if (idx == 0 || idx == 15) continue;
            auto px = [&](float gi) { return x0 + gi * dx; };
            auto py = [&](float gj) { return y0 + gj * dy; };
            Vec3 bottom(px(i + (iso - a) / (b - a)), py(j), z), right(px(i + 1), py(j + (iso - b) / (c - b)), z);
            Vec3 top(px(i + (iso - d) / (c - d)), py(j + 1), z), left(px(i), py(j + (iso - a) / (d - a)), z);
            switch (idx) {
                case 1: case 14: segs.push_back({left, bottom}); break;
                case 2: case 13: segs.push_back({bottom, right}); break;
                case 3: case 12: segs.push_back({left, right}); break;
                case 4: case 11: segs.push_back({right, top}); break;
                case 6: case 9: segs.push_back({bottom, top}); break;
                case 7: case 8: segs.push_back({left, top}); break;
                case 5: segs.push_back({left, top}); segs.push_back({bottom, right}); break;
                case 10: segs.push_back({left, bottom}); segs.push_back({right, top}); break;
                default: break;
            }
        }
    return segs;
}

// ---------------------------------------------------------------------------------------------
// Grand tour
// ---------------------------------------------------------------------------------------------

void gram_schmidt(std::vector<float> &cols, int d, int ncols) {
    for (int c = 0; c < ncols; ++c) {
        float *col = &cols[c * d];
        for (int p = 0; p < c; ++p) {
            const float *prev = &cols[p * d];
            float s = 0;
            for (int j = 0; j < d; ++j) s += col[j] * prev[j];
            for (int j = 0; j < d; ++j) col[j] -= s * prev[j];
        }
        float n = 0;
        for (int j = 0; j < d; ++j) n += col[j] * col[j];
        n = std::sqrt(n);
        if (n < 1e-12f) {  // degenerate: fall back to a basis vector
            for (int j = 0; j < d; ++j) col[j] = (j == c % d) ? 1.0f : 0.0f;
            continue;
        }
        for (int j = 0; j < d; ++j) col[j] /= n;
    }
}

void Tour::frame_at(float t, std::vector<float> &out) const {
    int nk = count();
    out.assign(static_cast<size_t>(3) * d, 0.0f);
    if (nk == 0) return;
    float tt = std::fmod(t, static_cast<float>(nk));
    if (tt < 0) tt += nk;
    int k0 = static_cast<int>(tt) % nk, k1 = (k0 + 1) % nk;
    float s = ease::smooth(tt - std::floor(tt));
    const float *A = &keyframes[static_cast<size_t>(k0) * 3 * d];
    const float *B = &keyframes[static_cast<size_t>(k1) * 3 * d];
    for (int i = 0; i < 3 * d; ++i) out[i] = (1 - s) * A[i] + s * B[i];
    gram_schmidt(out, d, 3);
}

Vec3 project_row(const std::vector<float> &frame, int d, const float *x) {
    float c[3] = {0, 0, 0};
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < d; ++j) c[k] += frame[k * d + j] * x[j];
    return {c[0], c[1], c[2]};
}
