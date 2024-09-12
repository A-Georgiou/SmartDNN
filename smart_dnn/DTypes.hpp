#ifndef DTYPES_HPP
#define DTYPES_HPP

#include <unordered_map>
#include <typeinfo>
#include <string>

namespace sdnn {
    enum class dtype {
        f16 = 0, // 16-bit float
        f32 = 1, // 32-bit float
        f64 = 2, // 64-bit float
        b8 = 3, // 8-bit boolean
        s16 = 4, // 16-bit signed integer
        s32 = 5, // 32-bit signed integer
        s64 = 6, // 64-bit signed integer
        u8 = 7, // 8-bit unsigned integer
        u16 = 8, // 16-bit unsigned integer
        u32 = 9, // 32-bit unsigned integer
        u64 = 10 // 64-bit unsigned integer
    };

    template<typename T>
    struct dtype_trait;

    template<> struct dtype_trait<float> { static constexpr dtype value = dtype::f32; };
    template<> struct dtype_trait<double> { static constexpr dtype value = dtype::f64; };
    template<> struct dtype_trait<int> { static constexpr dtype value = dtype::s32; };
    template<> struct dtype_trait<unsigned> { static constexpr dtype value = dtype::u32; };
    template<> struct dtype_trait<char> { static constexpr dtype value = dtype::b8; };
    template<> struct dtype_trait<unsigned char> { static constexpr dtype value = dtype::u8; };
    template<> struct dtype_trait<long> { static constexpr dtype value = dtype::s64; };
    template<> struct dtype_trait<unsigned long> { static constexpr dtype value = dtype::u64; };
    template<> struct dtype_trait<long long> { static constexpr dtype value = dtype::s64; };
    template<> struct dtype_trait<unsigned long long> { static constexpr dtype value = dtype::u64; };
    template<> struct dtype_trait<bool> { static constexpr dtype value = dtype::b8; };
    template<> struct dtype_trait<short> { static constexpr dtype value = dtype::s16; };
    template<> struct dtype_trait<unsigned short> { static constexpr dtype value = dtype::u16; };

    template<dtype DType>
    struct cpp_type;

    template<> struct cpp_type<dtype::f32> { using type = float;};
    template<> struct cpp_type<dtype::f64> { using type = double;};
    template<> struct cpp_type<dtype::b8> { using type = bool; };
    template<> struct cpp_type<dtype::s16> { using type = short; };
    template<> struct cpp_type<dtype::s32> { using type = int; };
    template<> struct cpp_type<dtype::s64> { using type = long; };
    template<> struct cpp_type<dtype::u8> { using type = unsigned char; };
    template<> struct cpp_type<dtype::u16> { using type = unsigned short; };
    template<> struct cpp_type<dtype::u32> { using type = unsigned; };
    template<> struct cpp_type<dtype::u64> { using type = unsigned long; };

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