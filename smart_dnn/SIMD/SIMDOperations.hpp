#ifndef SIMD_OPERATIONS_HPP
#define SIMD_OPERATIONS_HPP

#include "SIMDTypes.hpp"
#include "SIMDSelection.hpp"
#include "../Tensor.hpp"
template<typename SIMDType>
struct SIMDOperations {
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
    static void sub(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] - b.getData()[i];
        }
    }
    static void mul(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] * b.getData()[i];
        }
    }
    static void div(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); ++i) {
            result.getData()[i] = a.getData()[i] / b.getData()[i];
        }
    }
};

// Specialization for AVX512
#if defined(__AVX512F__)
template<>
struct SIMDOperations<AVX512> {
    static void add(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 16) {
            _mm512_storeu_ps(&result.getData()[i], _mm512_add_ps(_mm512_loadu_ps(&a.getData()[i]), _mm512_loadu_ps(&b.getData()[i])));
        }
    }
    static void sub(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 16) {
            _mm512_storeu_ps(&result.getData()[i], _mm512_sub_ps(_mm512_loadu_ps(&a.getData()[i]), _mm512_loadu_ps(&b.getData()[i])));
        }
    }
    static void mul(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 16) {
            _mm512_storeu_ps(&result.getData()[i], _mm512_mul_ps(_mm512_loadu_ps(&a.getData()[i]), _mm512_loadu_ps(&b.getData()[i])));
        }
    }
    static void div(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 16) {
            _mm512_storeu_ps(&result.getData()[i], _mm512_div_ps(_mm512_loadu_ps(&a.getData()[i]), _mm512_loadu_ps(&b.getData()[i])));
        }
    }
};
#endif // __AVX512F__

// Specialization for AVX2
#if defined(__AVX2__)
template<>
struct SIMDOperations<AVX2> {
    static void add(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 8) {
            _mm256_storeu_ps(&result.getData()[i], _mm256_add_ps(_mm256_loadu_ps(&a.getData()[i]), _mm256_loadu_ps(&b.getData()[i])));
        }
    }
    static void sub(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 8) {
            _mm256_storeu_ps(&result.getData()[i], _mm256_sub_ps(_mm256_loadu_ps(&a.getData()[i]), _mm256_loadu_ps(&b.getData()[i])));
        }
    }
    static void mul(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 8) {
            _mm256_storeu_ps(&result.getData()[i], _mm256_mul_ps(_mm256_loadu_ps(&a.getData()[i]), _mm256_loadu_ps(&b.getData()[i])));
        }
    }
    static void div(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 8) {
            _mm256_storeu_ps(&result.getData()[i], _mm256_div_ps(_mm256_loadu_ps(&a.getData()[i]), _mm256_loadu_ps(&b.getData()[i])));
        }
    }
};
#endif // __AVX2__

// Specialization for SSE4.2
#if defined(__SSE4_2__)
template<>
struct SIMDOperations<SSE4_2> {
    static void add(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 4) {
            _mm_storeu_ps(&result.getData()[i], _mm_add_ps(_mm_loadu_ps(&a.getData()[i]), _mm_loadu_ps(&b.getData()[i])));
        }
    }
    static void sub(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 4) {
            _mm_storeu_ps(&result.getData()[i], _mm_sub_ps(_mm_loadu_ps(&a.getData()[i]), _mm_loadu_ps(&b.getData()[i])));
        }
    }
    static void mul(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 4) {
            _mm_storeu_ps(&result.getData()[i], _mm_mul_ps(_mm_loadu_ps(&a.getData()[i]), _mm_loadu_ps(&b.getData()[i])));
        }
    }
    static void div(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 4) {
            _mm_storeu_ps(&result.getData()[i], _mm_div_ps(_mm_loadu_ps(&a.getData()[i]), _mm_loadu_ps(&b.getData()[i])));
        }
    }
};
#endif // __SSE4_2__

// Specialization for NEON (ARM)
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
template<>
struct SIMDOperations<NEON> {

    static void add(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 4) {
            vst1q_f32(&result.getData()[i], vaddq_f32(vld1q_f32(&a.getData()[i]), vld1q_f32(&b.getData()[i])));
        }
    }
    static void sub(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 4) {
            vst1q_f32(&result.getData()[i], vsubq_f32(vld1q_f32(&a.getData()[i]), vld1q_f32(&b.getData()[i])));
        }
    }
    static void mul(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 4) {
            vst1q_f32(&result.getData()[i], vmulq_f32(vld1q_f32(&a.getData()[i]), vld1q_f32(&b.getData()[i])));
        }
    }
    static void div(const Tensor& a, const Tensor& b, Tensor& result) {
        for (size_t i = 0; i < a.getData().size(); i += 4) {
            vst1q_f32(&result.getData()[i], vdivq_f32(vld1q_f32(&a.getData()[i]), vld1q_f32(&b.getData()[i])));
        }
    }
};

#endif // __ARM_NEON

#endif // SIMD_OPERATIONS_HPP
