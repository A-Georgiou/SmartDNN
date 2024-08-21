#ifndef SIMD_OPERATIONS_HPP
#define SIMD_OPERATIONS_HPP

#include "SIMDTypes.hpp"
#include "SIMDSelection.hpp"
#include "../Tensor.hpp"

template<typename SIMDType>
struct SIMDOperations {
    static constexpr size_t width();
    static void add(const Tensor& a, const Tensor& b, Tensor& result);
    static void sub(const Tensor& a, const Tensor& b, Tensor& result);
    static void mul(const Tensor& a, const Tensor& b, Tensor& result);
    static void div(const Tensor& a, const Tensor& b, Tensor& result);
};

// Specialization for no SIMD
template<>
struct SIMDOperations<NoSIMD> {

    static void add(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] + b.getData()[i];
        }
    }

    static void add(const Tensor& a, float scalar, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] + scalar;
        }
    }
    
    static void sub(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] - b.getData()[i];
        }
    }

    static void sub(const Tensor& a, float scalar, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] - scalar;
        }
    }

    static void mul(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] * b.getData()[i];
        }
    }

    static void mul(const Tensor& a, float scalar, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] * scalar;
        }
    }

    static void div(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); ++i) {
            result[i] = a[i] / b[i];
        }
    }

    static void div(const Tensor& a, float scalar, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); ++i) {
            result[i] = a[i] / scalar;
        }
    }

    static void sqrt(const Tensor& a, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); ++i) {
            result[i] = std::sqrt(a[i]);
        }
    }

};

// Specialization for AVX512
#if defined(__AVX512F__)
template<>
struct SIMDOperations<AVX512> {
    static constexpr size_t width() { return 16; }

    static void add(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m512 va = _mm512_loadu_ps(&a.getData()[i]);
            __m512 vb = _mm512_loadu_ps(&b.getData()[i]);
            __m512 vr = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] + b.getData()[i];
        }
    }

    static void add(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        __m512 scalar16 = _mm512_set1_ps(scalar);
        for (; i < a.getData().size(); i += width()) {
            __m512 va = _mm512_loadu_ps(&a.getData()[i]);
            __m512 vr = _mm512_add_ps(va, scalar16);
            _mm512_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] + scalar;
        }
    }

    static void sub(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m512 va = _mm512_loadu_ps(&a.getData()[i]);
            __m512 vb = _mm512_loadu_ps(&b.getData()[i]);
            __m512 vr = _mm512_sub_ps(va, vb);
            _mm512_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] - b.getData()[i];
        }
    }

    static void sub(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        __m512 scalar16 = _mm512_set1_ps(scalar);
        for (; i < a.getData().size(); i += width()) {
            __m512 va = _mm512_loadu_ps(&a.getData()[i]);
            __m512 vr = _mm512_sub_ps(va, scalar16);
            _mm512_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] - scalar;
        }
    }

    static void mul(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m512 va = _mm512_loadu_ps(&a.getData()[i]);
            __m512 vb = _mm512_loadu_ps(&b.getData()[i]);
            __m512 vr = _mm512_mul_ps(va, vb);
            _mm512_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] * b.getData()[i];
        }
    }

    static void mul(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        __m512 scalar16 = _mm512_set1_ps(scalar);
        for (; i < a.getData().size(); i += width()) {
            __m512 va = _mm512_loadu_ps(&a.getData()[i]);
            __m512 vr = _mm512_mul_ps(va, scalar16);
            _mm512_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] * scalar;
        }
    }

    static void div(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m512 va = _mm512_loadu_ps(&a.getData()[i]);
            __m512 vb = _mm512_loadu_ps(&b.getData()[i]);
            __m512 vr = _mm512_div_ps(va, vb);
            _mm512_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] / b.getData()[i];
        }
    }

    static void div(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        __m512 scalar16 = _mm512_set1_ps(scalar);
        for (; i < a.getData().size(); i += width()) {
            __m512 va = _mm512_loadu_ps(&a.getData()[i]);
            __m512 vr = _mm512_div_ps(va, scalar16);
            _mm512_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] / scalar;
        }
    }

    static void sqrt(const Tensor& a, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m512 va = _mm512_loadu_ps(&a.getData()[i]);
            __m512 vr = _mm512_sqrt_ps(va);
            _mm512_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = std::sqrt(a.getData()[i]);
        }
    }
};
#endif // __AVX512F__


// Specialization for AVX2
#if defined(__AVX2__)
template<>
struct SIMDOperations<AVX2> {
    static constexpr size_t width() { return 8; }

    static void add(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m256 va = _mm256_loadu_ps(&a.getData()[i]);
            __m256 vb = _mm256_loadu_ps(&b.getData()[i]);
            __m256 vr = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] + b.getData()[i];
        }
    }

    static void add(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        __m256 scalar8 = _mm256_set1_ps(scalar);
        for (; i < a.getData().size(); i += width()) {
            __m256 va = _mm256_loadu_ps(&a.getData()[i]);
            __m256 vr = _mm256_add_ps(va, scalar8);
            _mm256_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] + scalar;
        }
    }

    static void sub(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m256 va = _mm256_loadu_ps(&a.getData()[i]);
            __m256 vb = _mm256_loadu_ps(&b.getData()[i]);
            __m256 vr = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] - b.getData()[i];
        }
    }

    static void sub(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        __m256 scalar8 = _mm256_set1_ps(scalar);
        for (; i < a.getData().size(); i += width()) {
            __m256 va = _mm256_loadu_ps(&a.getData()[i]);
            __m256 vr = _mm256_sub_ps(va, scalar8);
            _mm256_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] - scalar;
        }
    }

    static void mul(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m256 va = _mm256_loadu_ps(&a.getData()[i]);
            __m256 vb = _mm256_loadu_ps(&b.getData()[i]);
            __m256 vr = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] * b.getData()[i];
        }
    }

    static void mul(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        __m256 scalar8 = _mm256_set1_ps(scalar);
        for (; i < a.getData().size(); i += width()) {
            __m256 va = _mm256_loadu_ps(&a.getData()[i]);
            __m256 vr = _mm256_mul_ps(va, scalar8);
            _mm256_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] * scalar;
        }
    }

    static void div(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m256 va = _mm256_loadu_ps(&a.getData()[i]);
            __m256 vb = _mm256_loadu_ps(&b.getData()[i]);
            __m256 vr = _mm256_div_ps(va, vb);
            _mm256_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] / b.getData()[i];
        }
    }

    static void div(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        __m256 scalar8 = _mm256_set1_ps(scalar);
        for (; i < a.getData().size(); i += width()) {
            __m256 va = _mm256_loadu_ps(&a.getData()[i]);
            __m256 vr = _mm256_div_ps(va, scalar8);
            _mm256_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] / scalar;
        }
    }

    static void sqrt(const Tensor& a, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m256 va = _mm256_loadu_ps(&a.getData()[i]);
            __m256 vr = _mm256_sqrt_ps(va);
            _mm256_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = std::sqrt(a.getData()[i]);
        }
    }
};
#endif // __AVX2__

// Specialization for SSE4.2
#if defined(__SSE4_2__)
template<>
struct SIMDOperations<SSE4_2> {
    static constexpr size_t width() { return 4; }

    static void add(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m128 va = _mm_loadu_ps(&a.getData()[i]);
            __m128 vb = _mm_loadu_ps(&b.getData()[i]);
            __m128 vr = _mm_add_ps(va, vb);
            _mm_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] + b.getData()[i];
        }
    }

    static void add(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        __m128 scalar4 = _mm_set1_ps(scalar);
        for (; i < a.getData().size(); i += width()) {
            __m128 va = _mm_loadu_ps(&a.getData()[i]);
            __m128 vr = _mm_add_ps(va, scalar4);
            _mm_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] + scalar;
        }
    }

    static void sub(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m128 va = _mm_loadu_ps(&a.getData()[i]);
            __m128 vb = _mm_loadu_ps(&b.getData()[i]);
            __m128 vr = _mm_sub_ps(va, vb);
            _mm_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] - b.getData()[i];
        }
    }

    static void sub(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        __m128 scalar4 = _mm_set1_ps(scalar);
        for (; i < a.getData().size(); i += width()) {
            __m128 va = _mm_loadu_ps(&a.getData()[i]);
            __m128 vr = _mm_sub_ps(va, scalar4);
            _mm_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] - scalar;
        }
    }

    static void mul(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m128 va = _mm_loadu_ps(&a.getData()[i]);
            __m128 vb = _mm_loadu_ps(&b.getData()[i]);
            __m128 vr = _mm_mul_ps(va, vb);
            _mm_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] * b.getData()[i];
        }
    }

    static void mul(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        __m128 scalar4 = _mm_set1_ps(scalar);
        for (; i < a.getData().size(); i += width()) {
            __m128 va = _mm_loadu_ps(&a.getData()[i]);
            __m128 vr = _mm_mul_ps(va, scalar4);
            _mm_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] * scalar;
        }
    }

    static void div(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m128 va = _mm_loadu_ps(&a.getData()[i]);
            __m128 vb = _mm_loadu_ps(&b.getData()[i]);
            __m128 vr = _mm_div_ps(va, vb);
            _mm_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] / b.getData()[i];
        }
    }

    static void div(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        __m128 scalar4 = _mm_set1_ps(scalar);
        for (; i < a.getData().size(); i += width()) {
            __m128 va = _mm_loadu_ps(&a.getData()[i]);
            __m128 vr = _mm_div_ps(va, scalar4);
            _mm_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] / scalar;
        }
    }

    static void sqrt(const Tensor& a, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            __m128 va = _mm_loadu_ps(&a.getData()[i]);
            __m128 vr = _mm_sqrt_ps(va);
            _mm_storeu_ps(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = std::sqrt(a.getData()[i]);
        }
    }
};
#endif // __SSE4_2__


// Specialization for NEON (ARM)
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
template<>
struct SIMDOperations<NEON> {
    static constexpr size_t width() { return 4; }

    static void add(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            float32x4_t va = vld1q_f32(&a.getData()[i]);
            float32x4_t vb = vld1q_f32(&b.getData()[i]);
            float32x4_t vr = vaddq_f32(va, vb);
            vst1q_f32(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] + b.getData()[i];
        }
    }

    static void add(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        float32x4_t scalar4 = vdupq_n_f32(scalar);
        for (; i < a.getData().size(); i += width()) {
            float32x4_t va = vld1q_f32(&a.getData()[i]);
            float32x4_t vr = vaddq_f32(va, scalar4);
            vst1q_f32(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] + scalar;
        }
    }

    static void sub(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            float32x4_t va = vld1q_f32(&a.getData()[i]);
            float32x4_t vb = vld1q_f32(&b.getData()[i]);
            float32x4_t vr = vsubq_f32(va, vb);
            vst1q_f32(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] - b.getData()[i];
        }
    }

    static void sub(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        float32x4_t scalar4 = vdupq_n_f32(scalar);
        for (; i < a.getData().size(); i += width()) {
            float32x4_t va = vld1q_f32(&a.getData()[i]);
            float32x4_t vr = vsubq_f32(va, scalar4);
            vst1q_f32(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] - scalar;
        }
    }

    static void mul(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (size_t i = 0; i < a.getData().size(); i += width()) {
            float32x4_t va = vld1q_f32(&a.getData()[i]);
            float32x4_t vb = vld1q_f32(&b.getData()[i]);
            float32x4_t vr = vmulq_f32(va, vb);
            vst1q_f32(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] * b.getData()[i];
        }
    }

    static void mul(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        float32x4_t scalar4 = vdupq_n_f32(scalar);
        for (; i < a.getData().size(); i += width()) {
            float32x4_t va = vld1q_f32(&a.getData()[i]);
            float32x4_t vr = vmulq_f32(va, scalar4);
            vst1q_f32(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] * scalar;
        }
    }

    static void div(const Tensor& a, const Tensor& b, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            float32x4_t va = vld1q_f32(&a.getData()[i]);
            float32x4_t vb = vld1q_f32(&b.getData()[i]);
            float32x4_t reciprocal = vrecpeq_f32(vb);
            reciprocal = vmulq_f32(vrecpsq_f32(vb, reciprocal), reciprocal);
            float32x4_t vr = vmulq_f32(va, reciprocal);
            vst1q_f32(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] / b.getData()[i];
        }
    }

    static void div(const Tensor& a, float scalar, Tensor& result) {
        size_t i = 0;
        float32x4_t scalar4 = vdupq_n_f32(scalar);
        for (; i < a.getData().size(); i += width()) {
            float32x4_t va = vld1q_f32(&a.getData()[i]);
            float32x4_t reciprocal = vrecpeq_f32(scalar4);
            reciprocal = vmulq_f32(vrecpsq_f32(scalar4, reciprocal), reciprocal);
            float32x4_t vr = vmulq_f32(va, reciprocal);
            vst1q_f32(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] / scalar;
        }

    static void sqrt(const Tensor& a, Tensor& result) {
        size_t i = 0;
        for (; i < a.getData().size(); i += width()) {
            float32x4_t va = vld1q_f32(&a.getData()[i]);
            float32x4_t vr = vsqrtq_f32(va);
            vst1q_f32(&result.getData()[i], vr);
        }

        for (; i < a.getData().size(); ++i) {
            result.getData()[i] = std::sqrt(a.getData()[i]);
        }
    }
};
#endif // __ARM_NEON


#endif // SIMD_OPERATIONS_HPP