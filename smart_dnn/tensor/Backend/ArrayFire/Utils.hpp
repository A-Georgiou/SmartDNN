#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include "smart_dnn/DTypes.hpp"
#include "smart_dnn/Shape/Shape.hpp"
#include <af/data.h>

/*

    Set of helper utilities for ArrayFire Tensor implementation

*/


namespace sdnn {
    namespace utils {
    
        af::dtype sdnnToAfType(const sdnn::dtype type){
            static const std::unordered_map<sdnn::dtype, af::dtype> sdnnToAfTypeMap = {
                {sdnn::dtype::f16, af::dtype::f16},
                {sdnn::dtype::f32, af::dtype::f32},
                {sdnn::dtype::f64, af::dtype::f64},
                {sdnn::dtype::b8, af::dtype::b8},
                {sdnn::dtype::s16, af::dtype::s16},
                {sdnn::dtype::s32, af::dtype::s32},
                {sdnn::dtype::s64, af::dtype::s64},
                {sdnn::dtype::u8, af::dtype::u8},
                {sdnn::dtype::u16, af::dtype::u16},
                {sdnn::dtype::u32, af::dtype::u32},
                {sdnn::dtype::u64, af::dtype::u64}
            };
            return sdnnToAfTypeMap.at(type);
        }

        sdnn::dtype afToSdnnType(const af::dtype type){
            static const std::unordered_map<af::dtype, sdnn::dtype> afToSdnnTypeMap = {
                {af::dtype::f16, sdnn::dtype::f16},
                {af::dtype::f32, sdnn::dtype::f32},
                {af::dtype::f64, sdnn::dtype::f64},
                {af::dtype::b8, sdnn::dtype::b8},
                {af::dtype::s16, sdnn::dtype::s16},
                {af::dtype::s32, sdnn::dtype::s32},
                {af::dtype::s64, sdnn::dtype::s64},
                {af::dtype::u8, sdnn::dtype::u8},
                {af::dtype::u16, sdnn::dtype::u16},
                {af::dtype::u32, sdnn::dtype::u32},
                {af::dtype::u64, sdnn::dtype::u64}
            };
            return afToSdnnTypeMap.at(type);
        }

        af::dim4 shapeToAfDim(const Shape& shape){
            af::dim4 dim;
            for (size_t i = 0; i < shape.rank(); ++i) {
                dim[i] = shape[i];
            }
            return dim;
        }

        Shape afDimToShape(const af::dim4& dim){
            std::vector<int> dims(dim.ndims());
            for (size_t i = 0; i < dim.ndims(); ++i) {
                dims[i] = dim[i];
            }
            return Shape(dims);
        }

    } // namespace utils
} // namespace sdnn

#endif // UTILS_HPP