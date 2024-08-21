#ifndef SIMD_SELECTION_HPP
#define SIMD_SELECTION_HPP

#include "SIMDTypes.hpp"

// Include the SIMD types that match the architecture
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    // x86/64 Architecture
    #if defined(HAS_MAVX512F)
        #include <immintrin.h>
        using DefaultSIMD = AVX512;
    #elif defined(HAS_MAVX2)
        #include <immintrin.h>
        using DefaultSIMD = AVX2;
    #elif defined(HAS_MSSE4_2)
        #include <nmmintrin.h>
        using DefaultSIMD = SSE42;
    #else
        using DefaultSIMD = NoSIMD;
    #endif
#elif defined(__arm__) || defined(__aarch64__)
    // ARM Architecture (including Apple Silicon M1/M2)
    #include <arm_neon.h>
    using DefaultSIMD = NEON;
#else
    using DefaultSIMD = NoSIMD;
#endif

#endif // SIMD_SELECTION_HPP