#ifndef DTYPES_HPP
#define DTYPES_HPP

#include <unordered_map>
#include <typeinfo>
#include <string>

namespace sdnn {
    enum class dtype {
        f16 = 0, // 16-bit float (not standard C++ type)
        f32 = 1, // 32-bit float
        f64 = 2, // 64-bit float
        b8 = 3,  // 8-bit boolean
        s8 = 4,  // 8-bit signed integer
        s16 = 5, // 16-bit signed integer
        s32 = 6, // 32-bit signed integer
        s64 = 7, // 64-bit signed integer
        u8 = 8,  // 8-bit unsigned integer
        u16 = 9, // 16-bit unsigned integer
        u32 = 10, // 32-bit unsigned integer
        u64 = 11  // 64-bit unsigned integer
    };

    template<typename T>
    struct dtype_trait;

    // Primary type definitions
    template<> struct dtype_trait<float> { static constexpr dtype value = dtype::f32; };
    template<> struct dtype_trait<double> { static constexpr dtype value = dtype::f64; };
    template<> struct dtype_trait<bool> { static constexpr dtype value = dtype::b8; };
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
    template<> struct dtype_trait<long> { static constexpr dtype value = sizeof(long) == 4 ? dtype::s32 : dtype::s64; };
    template<> struct dtype_trait<unsigned long> { static constexpr dtype value = sizeof(unsigned long) == 4 ? dtype::u32 : dtype::u64; };

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

   template<dtype DType>
    struct cpp_type;

    template<> struct cpp_type<dtype::f32> { using type = float; };
    template<> struct cpp_type<dtype::f64> { using type = double; };
    template<> struct cpp_type<dtype::b8> { using type = bool; };
    template<> struct cpp_type<dtype::s8> { using type = int8_t; };
    template<> struct cpp_type<dtype::s16> { using type = int16_t; };
    template<> struct cpp_type<dtype::s32> { using type = int32_t; };
    template<> struct cpp_type<dtype::s64> { using type = int64_t; };
    template<> struct cpp_type<dtype::u8> { using type = uint8_t; };
    template<> struct cpp_type<dtype::u16> { using type = uint16_t; };
    template<> struct cpp_type<dtype::u32> { using type = uint32_t; };
    template<> struct cpp_type<dtype::u64> { using type = uint64_t; };

    template <typename T>
    constexpr T* dtype_cast(void* data, dtype dtype) {
        switch (dtype) {
            case dtype::f32: return static_cast<T*>(static_cast<float*>(data));
            case dtype::f64: return static_cast<T*>(static_cast<double*>(data));
            case dtype::b8: return static_cast<T*>(static_cast<bool*>(data));
            case dtype::s8: return static_cast<T*>(static_cast<int8_t*>(data));
            case dtype::s16: return static_cast<T*>(static_cast<int16_t*>(data));
            case dtype::s32: return static_cast<T*>(static_cast<int32_t*>(data));
            case dtype::s64: return static_cast<T*>(static_cast<int64_t*>(data));
            case dtype::u8: return static_cast<T*>(static_cast<uint8_t*>(data));
            case dtype::u16: return static_cast<T*>(static_cast<uint16_t*>(data));
            case dtype::u32: return static_cast<T*>(static_cast<uint32_t*>(data));
            case dtype::u64: return static_cast<T*>(static_cast<uint64_t*>(data));
            default: throw std::runtime_error("Unknown DataType");
        }
    }

    constexpr size_t dtype_size(dtype dt) {
        switch (dt) {
            case dtype::f32: return sizeof(float);
            case dtype::f64: return sizeof(double);
            case dtype::b8: return sizeof(bool);
            case dtype::s8: return sizeof(int8_t);
            case dtype::s16: return sizeof(int16_t);
            case dtype::s32: return sizeof(int32_t);
            case dtype::s64: return sizeof(int64_t);
            case dtype::u8: return sizeof(uint8_t);
            case dtype::u16: return sizeof(uint16_t);
            case dtype::u32: return sizeof(uint32_t);
            case dtype::u64: return sizeof(uint64_t);
            default: throw std::runtime_error("Unknown DataType");
        }
    }

    inline std::string dtypeToString(dtype dt) {
        switch (dt) {
            case dtype::f32: return "f32";
            case dtype::f64: return "f64";
            case dtype::b8: return "b8";
            case dtype::s8: return "s8";
            case dtype::s16: return "s16";
            case dtype::s32: return "s32";
            case dtype::s64: return "s64";
            case dtype::u8: return "u8";
            case dtype::u16: return "u16";
            case dtype::u32: return "u32";
            case dtype::u64: return "u64";
            default: throw std::runtime_error("Unknown DataType");
        }
    }

    namespace detail {
        template<dtype... Ts>
        struct type_list {};

        template<typename T, typename... Ts>
        struct concat;

        template<dtype... Ts, dtype... Us>
        struct concat<type_list<Ts...>, type_list<Us...>> {
            using type = type_list<Ts..., Us...>;
        };

        template<typename T>
        struct to_unique_ptr {
            using type = std::unique_ptr<T[]>;
        };

        template<dtype T>
        using to_unique_ptr_t = typename to_unique_ptr<typename cpp_type<T>::type>::type;

        using supported_types = type_list<dtype::f32, dtype::f64, dtype::s16, dtype::s32, dtype::s64, 
                                        dtype::u8, dtype::u16, dtype::u32, dtype::u64>;

        template<typename T>
        struct variant_from_type_list;

        template<dtype... Ts>
        struct variant_from_type_list<type_list<Ts...>> {
            using type = std::variant<to_unique_ptr_t<Ts>...>;
        };
    }
}

#endif // DTYPES_HPP