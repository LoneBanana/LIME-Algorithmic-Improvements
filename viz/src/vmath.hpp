// Minimal vector / matrix maths for the renderer (no external libraries).
// Matrices are column-major, as OpenGL expects.
#pragma once

#include <cmath>

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(float s, Vec3 a) { return a * s; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(Vec3 a) {
    float l = length(a);
    return l > 1e-20f ? a * (1.0f / l) : Vec3(0, 0, 1);
}
inline Vec3 lerp(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }

struct Vec4 {
    float x = 0, y = 0, z = 0, w = 0;
};

struct Mat4 {
    float m[16];  // column-major: m[col * 4 + row]

    static Mat4 identity() {
        Mat4 r{};
        for (int i = 0; i < 16; ++i) r.m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        return r;
    }

    static Mat4 perspective(float fovy, float aspect, float znear, float zfar) {
        Mat4 r{};
        float f = 1.0f / std::tan(fovy * 0.5f);
        for (float &v : r.m) v = 0;
        r.m[0] = f / aspect;
        r.m[5] = f;
        r.m[10] = (zfar + znear) / (znear - zfar);
        r.m[11] = -1.0f;
        r.m[14] = 2.0f * zfar * znear / (znear - zfar);
        return r;
    }

    static Mat4 ortho(float l, float r_, float b, float t, float n, float f) {
        Mat4 r = identity();
        r.m[0] = 2.0f / (r_ - l);
        r.m[5] = 2.0f / (t - b);
        r.m[10] = -2.0f / (f - n);
        r.m[12] = -(r_ + l) / (r_ - l);
        r.m[13] = -(t + b) / (t - b);
        r.m[14] = -(f + n) / (f - n);
        return r;
    }

    static Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up) {
        Vec3 f = normalize(target - eye);
        Vec3 s = normalize(cross(f, up));
        Vec3 u = cross(s, f);
        Mat4 r = identity();
        r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
        r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
        r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
        r.m[12] = -dot(s, eye);
        r.m[13] = -dot(u, eye);
        r.m[14] = dot(f, eye);
        return r;
    }

    Mat4 operator*(const Mat4 &b) const {
        Mat4 r{};
        for (int c = 0; c < 4; ++c)
            for (int rr = 0; rr < 4; ++rr) {
                float s = 0;
                for (int k = 0; k < 4; ++k) s += m[k * 4 + rr] * b.m[c * 4 + k];
                r.m[c * 4 + rr] = s;
            }
        return r;
    }

    Vec4 operator*(const Vec4 &v) const {
        return {m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12] * v.w, m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13] * v.w,
                m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w, m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w};
    }
};

// manim's rate functions (manim/utils/rate_functions.py), re-implemented.
namespace ease {
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float smooth(float t, float inflection = 10.0f) {
    float err = sigmoid(-inflection / 2.0f);
    float v = (sigmoid(inflection * (t - 0.5f)) - err) / (1.0f - 2.0f * err);
    return v < 0 ? 0 : (v > 1 ? 1 : v);
}
inline float there_and_back(float t) { return smooth(t < 0.5f ? 2.0f * t : 2.0f * (1.0f - t)); }
inline float ease_in_out_cubic(float t) { return t < 0.5f ? 4 * t * t * t : 1 - std::pow(-2 * t + 2, 3.0f) / 2; }
inline float apply(int kind, float t) {
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    switch (kind) {
        case 1: return smooth(t);
        case 2: return ease_in_out_cubic(t);
        case 3: return there_and_back(t);
        default: return t;
    }
}
}  // namespace ease
