#ifndef DTYPES_HPP
#define DTYPES_HPP

#include <unordered_map>
#include <typeinfo>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <type_traits>

namespace sdnn {
    enum class dtype {
        f16 = 0, // 16-bit float (not standard C++ type)
        f32 = 1, // 32-bit float
        f64 = 2, // 64-bit float
        s8 = 3,  // 8-bit signed integer
        s16 = 4, // 16-bit signed integer
        s32 = 5, // 32-bit signed integer
        s64 = 6, // 64-bit signed integer
        u8 = 7,  // 8-bit unsigned integer
        u16 = 8, // 16-bit unsigned integer
        u32 = 9, // 32-bit unsigned integer
        u64 = 10, // 64-bit unsigned integer
        b8 = 11, // 8-bit boolean
        Undefined = 12
    };
    
    struct DataItem {
        void* data;
        dtype type;
    };

    struct DataContainer {
        void* data;
        dtype type;
        size_t num_elements;
    };

    template<typename T>
    struct dtype_trait;

    // Primary type definitions
    template<> struct dtype_trait<bool> { static constexpr dtype value = dtype::b8; };
    template<> struct dtype_trait<float> { static constexpr dtype value = dtype::f32; };
    template<> struct dtype_trait<double> { static constexpr dtype value = dtype::f64; };
    template<> struct dtype_trait<int8_t> { static constexpr dtype value = dtype::s8; };
    template<> struct dtype_trait<int16_t> { static constexpr dtype value = dtype::s16; };
    template<> struct dtype_trait<int32_t> { static constexpr dtype value = dtype::s32; };
    template<> struct dtype_trait<int64_t> { static constexpr dtype value = dtype::s64; };
    template<> struct dtype_trait<uint8_t> { static constexpr dtype value = dtype::u8; };
    template<> struct dtype_trait<uint16_t> { static constexpr dtype value = dtype::u16; };
    template<> struct dtype_trait<uint32_t> { static constexpr dtype value = dtype::u32; };
    template<> struct dtype_trait<uint64_t> { static constexpr dtype value = dtype::u64; };
    // Additional definitions only if they differ from the primary types
    template<> struct dtype_trait<char> { static constexpr dtype value = std::is_signed<char>::value ? dtype::s8 : dtype::u8; };
    
    // Skip long long types that may conflict with int64_t/uint64_t on some systems
    // Code should use int64_t/uint64_t directly instead of long long

    template<typename T>
    constexpr T* safe_cast(void* data, dtype type) {
        using NonConstT = typename std::remove_const<T>::type;
        if (dtype_trait<NonConstT>::value != type) {
            throw std::runtime_error("Type mismatch in safe_cast");
        }
        return static_cast<T*>(data);
    }

    template<typename T>
    constexpr const T* safe_cast(const void* data, dtype type) {
        using NonConstT = typename std::remove_const<T>::type;
        if (dtype_trait<NonConstT>::value != type) {
            throw std::runtime_error("Type mismatch in safe_cast");
        }
        return static_cast<const T*>(data);
    }

    template<typename TypedOp>
    void applyTypedOperationHelper(dtype type, TypedOp op) {
        switch (type) {
            case dtype::f32: op(float{}); break;
            case dtype::f64: op(double{}); break;
            case dtype::s8: { signed char t{}; op(t); } break;  // Map int8_t to signed char
            case dtype::s16: op(short{}); break;  // Map int16_t to short
            case dtype::s32: op(int{}); break;  // Map int32_t to int
            case dtype::s64: op(long{}); break; // Map int64_t to long (64-bit on most systems)
            case dtype::u8: { unsigned char t{}; op(t); } break;
            case dtype::u16: { unsigned short t{}; op(t); } break;  // Map uint16_t to unsigned short
            case dtype::u32: { unsigned int t{}; op(t); } break;  // Map uint32_t to unsigned int
            case dtype::u64: { unsigned long t{}; op(t); } break; // Map uint64_t to unsigned long
            case dtype::b8: op(bool{}); break;
            default: throw std::runtime_error("Unsupported dtype for operation");
        }
    }

    template<typename T1, typename T2>
    struct promoted_type {
        using type = decltype(std::declval<T1>() + std::declval<T2>());
    };

   template<dtype DType>
    struct cpp_type;

    template<> struct cpp_type<dtype::f32> { using type = float; };
    template<> struct cpp_type<dtype::f64> { using type = double; };
    template<> struct cpp_type<dtype::s8> { using type = int8_t; };
    template<> struct cpp_type<dtype::s16> { using type = int16_t; };
    template<> struct cpp_type<dtype::s32> { using type = int32_t; };
    template<> struct cpp_type<dtype::s64> { using type = int64_t; };
    template<> struct cpp_type<dtype::u8> { using type = uint8_t; };
    template<> struct cpp_type<dtype::u16> { using type = uint16_t; };
    template<> struct cpp_type<dtype::u32> { using type = uint32_t; };
    template<> struct cpp_type<dtype::u64> { using type = uint64_t; };
    template<> struct cpp_type<dtype::b8> { using type = uint8_t; };

    template <typename T>
    constexpr T dtype_cast(const void* data, dtype dtype) {
        if (data == nullptr) {
        throw std::runtime_error("Null pointer encountered in dtype_cast");
        }
        switch (dtype) {
            case dtype::f32: return static_cast<T>(*static_cast<const float*>(data));
            case dtype::f64: return static_cast<T>(*static_cast<const double*>(data));
            case dtype::s8:  return static_cast<T>(*static_cast<const int8_t*>(data));
            case dtype::s16: return static_cast<T>(*static_cast<const int16_t*>(data));
            case dtype::s32: return static_cast<T>(*static_cast<const int32_t*>(data));
            case dtype::s64: return static_cast<T>(*static_cast<const int64_t*>(data));
            case dtype::u8:  return static_cast<T>(*static_cast<const uint8_t*>(data));
            case dtype::u16: return static_cast<T>(*static_cast<const uint16_t*>(data));
            case dtype::u32: return static_cast<T>(*static_cast<const uint32_t*>(data));
            case dtype::u64: return static_cast<T>(*static_cast<const uint64_t*>(data));
            case dtype::b8: return static_cast<T>(*static_cast<const uint8_t*>(data));
            default: throw std::runtime_error("Unknown DataType");
        }
    }

    constexpr inline void* convert_dtype(void* dest, const void* src, dtype to_type, dtype from_type) {
        if (dest == nullptr || src == nullptr) {
            throw std::runtime_error("Null pointer encountered in convert_dtype");
        }
        switch(to_type) {
            case dtype::f32: *static_cast<float*>(dest) = dtype_cast<float>(src, from_type); break;
            case dtype::f64: *static_cast<double*>(dest) = dtype_cast<double>(src, from_type); break;
            case dtype::s8:  *static_cast<int8_t*>(dest) = dtype_cast<int8_t>(src, from_type); break;
            case dtype::s16: *static_cast<int16_t*>(dest) = dtype_cast<int16_t>(src, from_type); break;
            case dtype::s32: *static_cast<int32_t*>(dest) = dtype_cast<int32_t>(src, from_type); break;
            case dtype::s64: *static_cast<int64_t*>(dest) = dtype_cast<int64_t>(src, from_type); break;
            case dtype::u8:  *static_cast<uint8_t*>(dest) = dtype_cast<uint8_t>(src, from_type); break;
            case dtype::u16: *static_cast<uint16_t*>(dest) = dtype_cast<uint16_t>(src, from_type); break;
            case dtype::u32: *static_cast<uint32_t*>(dest) = dtype_cast<uint32_t>(src, from_type); break;
            case dtype::u64: *static_cast<uint64_t*>(dest) = dtype_cast<uint64_t>(src, from_type); break;
            case dtype::b8: *static_cast<uint8_t*>(dest) = dtype_cast<uint8_t>(src, from_type); break;
            default: throw std::runtime_error("Unknown destination type");
        }
        return dest;
    }

    constexpr size_t dtype_size(dtype dt) {
        switch (dt) {
            case dtype::f32: return sizeof(float);
            case dtype::f64: return sizeof(double);
            case dtype::s8: return sizeof(int8_t);
            case dtype::s16: return sizeof(int16_t);
            case dtype::s32: return sizeof(int32_t);
            case dtype::s64: return sizeof(int64_t);
            case dtype::u8: return sizeof(uint8_t);
            case dtype::u16: return sizeof(uint16_t);
            case dtype::u32: return sizeof(uint32_t);
            case dtype::u64: return sizeof(uint64_t);
            case dtype::b8: return sizeof(uint8_t);
            default: throw std::runtime_error("Unknown DataType");
        }
    }

    inline std::string dtypeToString(dtype dt) {
        switch (dt) {
            case dtype::f32: return "f32";
            case dtype::f64: return "f64";
            case dtype::s8: return "s8";
            case dtype::s16: return "s16";
            case dtype::s32: return "s32";
            case dtype::s64: return "s64";
            case dtype::u8: return "u8";
            case dtype::u16: return "u16";
            case dtype::u32: return "u32";
            case dtype::u64: return "u64";
            case dtype::b8: return "b8";
            default: throw std::runtime_error("Unknown DataType");
        }
    }

    constexpr bool is_floating_point(dtype dt) {
        return dt == dtype::f16 || dt == dtype::f32 || dt == dtype::f64;
    }

    constexpr bool is_signed(dtype dt) {
        return dt == dtype::s8 || dt == dtype::s16 || dt == dtype::s32 || dt == dtype::s64 ||
            is_floating_point(dt);
    }

    constexpr inline int type_rank(dtype dt) {
        switch (dt) {
            case dtype::b8: case dtype::u8: return 1;
            case dtype::s16: case dtype::u16: return 2;
            case dtype::s32: case dtype::u32: return 3;
            case dtype::s64: case dtype::u64: return 4;
            case dtype::f16: return 5;
            case dtype::f32: return 6;
            case dtype::f64: return 7;
            default: throw std::runtime_error("Unknown dtype in type_rank");
        }
    }

    constexpr inline dtype promotionOfTypes(dtype a, dtype b) {
        if (is_floating_point(a) || is_floating_point(b)) {
            dtype candidates[] = {a, b, dtype::f32};
            auto result = *std::max_element(candidates, candidates + 3, [](dtype x, dtype y) {
                return type_rank(x) < type_rank(y);
            });
            return result;
        }

        if (is_signed(a) == is_signed(b)) {
            return type_rank(a) >= type_rank(b) ? a : b;
        }

        dtype unsigned_type = is_signed(a) ? b : a;
        dtype signed_type = is_signed(a) ? a : b;

        if (type_rank(unsigned_type) >= type_rank(signed_type)) {
            return unsigned_type;
        } else {
            return signed_type;
        }
    }
}

#endif // DTYPES_HPP