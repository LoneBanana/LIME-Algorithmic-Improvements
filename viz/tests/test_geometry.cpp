// Native unit tests for the platform-independent geometry (run: viz/build.sh test).
#include <cmath>
#include <cstdio>
#include <vector>

#include "../src/geometry.hpp"

static int failures = 0;
#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                       \
            std::printf("\n");                              \
            ++failures;                                     \
        }                                                   \
    } while (0)

static float triangle_area(const MeshData &m) {
    double area = 0;
    for (size_t t = 0; t + 2 < m.idx.size(); t += 3) {
        const Vertex &a = m.v[m.idx[t]], &b = m.v[m.idx[t + 1]], &c = m.v[m.idx[t + 2]];
        Vec3 pa(a.pa[0], a.pa[1], a.pa[2]), pb(b.pa[0], b.pa[1], b.pa[2]), pc(c.pa[0], c.pa[1], c.pa[2]);
        area += 0.5 * length(cross(pb - pa, pc - pa));
    }
    return static_cast<float>(area);
}

static void test_sphere_isosurface() {
    const int n = 48;
    std::vector<float> f(n * n * n);
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                float x = -1 + 2.0f * i / (n - 1), y = -1 + 2.0f * j / (n - 1), z = -1 + 2.0f * k / (n - 1);
                f[(k * n + j) * n + i] = x * x + y * y + z * z;
            }
    const float r = 0.6f;
    MeshData m = marching_tetrahedra(f.data(), n, n, n, r * r, {-1, -1, -1}, {1, 1, 1}, {1, 1, 1, 1});
    CHECK(!m.idx.empty(), "sphere produced no triangles");
    float worst = 0, worst_normal = 1;
    for (const Vertex &v : m.v) {
        Vec3 p(v.pa[0], v.pa[1], v.pa[2]), nn(v.na[0], v.na[1], v.na[2]);
        worst = std::fmax(worst, std::fabs(length(p) - r));
        worst_normal = std::fmin(worst_normal, dot(normalize(p), nn));
    }
    float h = 2.0f / (n - 1);
    CHECK(worst < h * h, "vertex off the sphere by %g", worst);
    CHECK(worst_normal > 0.97f, "normal not radial: cos %g", worst_normal);
    float area = triangle_area(m), exact = 4 * 3.14159265f * r * r;
    CHECK(std::fabs(area - exact) / exact < 0.02f, "area %g vs %g", area, exact);
}

static void test_linear_field_is_exactly_planar() {
    const int n = 12;
    std::vector<float> f(n * n * n);
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                float x = -1 + 2.0f * i / (n - 1), y = -1 + 2.0f * j / (n - 1), z = -1 + 2.0f * k / (n - 1);
                f[(k * n + j) * n + i] = x + 2 * y - z;
            }
    MeshData m = marching_tetrahedra(f.data(), n, n, n, 0.3f, {-1, -1, -1}, {1, 1, 1}, {1, 1, 1, 1});
    CHECK(!m.idx.empty(), "plane produced no triangles");
    float worst = 0;
    for (const Vertex &v : m.v) worst = std::fmax(worst, std::fabs(v.pa[0] + 2 * v.pa[1] - v.pa[2] - 0.3f));
    CHECK(worst < 1e-5f, "plane residual %g", worst);
    for (size_t i = 0; i < m.idx.size(); ++i) CHECK(m.idx[i] < m.v.size(), "index out of range");
}

static void test_marching_squares_circle() {
    const int n = 64;
    std::vector<float> f(n * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            float x = -1 + 2.0f * i / (n - 1), y = -1 + 2.0f * j / (n - 1);
            f[j * n + i] = std::sqrt(x * x + y * y);
        }
    auto segs = marching_squares(f.data(), n, n, 0.5f, -1, 1, -1, 1, 0.25f);
    CHECK(segs.size() > 40, "too few segments: %zu", segs.size());
    float worst = 0, total = 0;
    for (auto &s : segs) {
        worst = std::fmax(worst, std::fabs(std::hypot(s.first.x, s.first.y) - 0.5f));
        CHECK(s.first.z == 0.25f && s.second.z == 0.25f, "segment not at z");
        total += length(s.second - s.first);
    }
    CHECK(worst < 2e-3f, "iso-line off circle by %g", worst);
    CHECK(std::fabs(total - 3.14159265f) < 0.01f, "perimeter %g", total);
}

static void test_tube_and_surface() {
    MeshData t = make_tube({{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1}}, 0.1f, 8, {1, 1, 1, 1});
    CHECK(t.v.size() == 4 * 8, "tube vertices %zu", t.v.size());
    CHECK(t.idx.size() == 3 * 8 * 6, "tube indices %zu", t.idx.size());
    float prev = -1;
    for (size_t i = 0; i < t.v.size(); i += 8) {
        CHECK(t.v[i].param >= prev, "reveal parameter not monotone");
        prev = t.v[i].param;
    }
    CHECK(std::fabs(prev - 1.0f) < 1e-6f, "last param %g", prev);

    const int nu = 5, nv = 4;
    std::vector<float> h(nu * nv, 0.5f), h2(nu * nv, 1.0f);
    h[7] = NAN;  // one hole: the four cells around node 7 disappear
    MeshData s = make_surface(h.data(), h2.data(), nu, nv, -1, 1, -1, 1, 2.0f, {1, 1, 1, 1}, false);
    CHECK(s.v.size() == static_cast<size_t>(nu * nv), "surface vertices");
    CHECK(s.idx.size() == static_cast<size_t>(((nu - 1) * (nv - 1) - 4) * 6), "surface indices %zu", s.idx.size());
    CHECK(s.v[0].pa[2] == 1.0f && s.v[0].pb[2] == 2.0f, "morph target heights");
    MeshData c = make_surface(h2.data(), nullptr, nu, nv, -1, 1, -1, 1, 1.0f, {1, 1, 1, 1}, true);
    CHECK(c.v.size() == static_cast<size_t>((nu - 1) * (nv - 1) * 4), "checker vertices");
}

static void test_icosphere() {
    MeshData s = make_icosphere(2);
    CHECK(s.v.size() == 162, "icosphere vertices %zu", s.v.size());
    CHECK(s.idx.size() == 320 * 3, "icosphere triangles %zu", s.idx.size() / 3);
    for (const Vertex &v : s.v) CHECK(std::fabs(length(Vec3(v.pa[0], v.pa[1], v.pa[2])) - 1) < 1e-5f, "not unit");
}

static void test_tour_frames_orthonormal() {
    const int d = 7;
    Tour tour;
    tour.d = d;
    std::vector<float> frame(3 * d);
    unsigned seed = 12345;
    for (int k = 0; k < 4; ++k) {
        for (float &v : frame) {
            seed = seed * 1664525u + 1013904223u;
            v = (seed >> 8) / 16777216.0f - 0.5f;
        }
        gram_schmidt(frame, d, 3);
        tour.keyframes.insert(tour.keyframes.end(), frame.begin(), frame.end());
    }
    std::vector<float> out;
    for (float t = 0; t < 4; t += 0.37f) {
        tour.frame_at(t, out);
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                float s = 0;
                for (int j = 0; j < d; ++j) s += out[a * d + j] * out[b * d + j];
                CHECK(std::fabs(s - (a == b ? 1.0f : 0.0f)) < 1e-5f, "frame not orthonormal at t=%g", t);
            }
    }
    tour.frame_at(2.0f, out);
    for (int i = 0; i < 3 * d; ++i) CHECK(std::fabs(out[i] - tour.keyframes[2 * 3 * d + i]) < 1e-5f, "keyframe 2");
    std::vector<float> x(d, 0.0f);
    x[0] = 1.0f;
    Vec3 p = project_row(out, d, x.data());
    CHECK(std::fabs(p.x - out[0]) < 1e-6f && std::fabs(p.y - out[d]) < 1e-6f, "projection");
}

static void test_easing() {
    CHECK(std::fabs(ease::smooth(0)) < 1e-6f && std::fabs(ease::smooth(1) - 1) < 1e-6f, "smooth endpoints");
    CHECK(std::fabs(ease::smooth(0.5f) - 0.5f) < 1e-6f, "smooth midpoint");
    CHECK(std::fabs(ease::there_and_back(0.5f) - 1) < 1e-6f && ease::there_and_back(1) < 1e-6f, "there and back");
}

int main() {
    test_sphere_isosurface();
    test_linear_field_is_exactly_planar();
    test_marching_squares_circle();
    test_tube_and_surface();
    test_icosphere();
    test_tour_frames_orthonormal();
    test_easing();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all geometry tests passed\n");
    return 0;
}
