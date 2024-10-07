#include <typeindex>
#include <unordered_map>
#include "smart_dnn/DTypes.hpp"

namespace sdnn {

    // Function to map fl::dtype to primitive types
// Mapping from dtype to corresponding primitive types
    std::type_index sdnnTypeToPrimitive(dtype type) {
        static const std::unordered_map<dtype, std::type_index> kDtypeToPrimitive = {
            {dtype::f16, std::type_index(typeid(float))},  // Temporarily assuming f16 -> float
            {dtype::f32, std::type_index(typeid(float))},
            {dtype::f64, std::type_index(typeid(double))},
            {dtype::s16, std::type_index(typeid(short))},
            {dtype::s32, std::type_index(typeid(int))},
            {dtype::s64, std::type_index(typeid(long))},
            {dtype::u8, std::type_index(typeid(unsigned char))},
            {dtype::u16, std::type_index(typeid(unsigned short))},
            {dtype::u32, std::type_index(typeid(unsigned int))},
            {dtype::u64, std::type_index(typeid(unsigned long))}
        };

        auto it = kDtypeToPrimitive.find(type);
        if (it != kDtypeToPrimitive.end()) {
            return it->second;
        } else {
            throw std::invalid_argument("Unsupported dtype");
        }
    }

} // namespace sdnn