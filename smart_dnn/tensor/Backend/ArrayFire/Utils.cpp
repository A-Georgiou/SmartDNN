#ifndef UTILS_CPP
#define UTILS_CPP

#include "smart_dnn/tensor/Backend/ArrayFire/Utils.hpp"
namespace sdnn {
    namespace utils {
    
        af::dtype sdnnToAfType(const sdnn::dtype type){
            static const std::unordered_map<sdnn::dtype, af::dtype> sdnnToAfTypeMap = {
                {sdnn::dtype::f16, af::dtype::f16},
                {sdnn::dtype::f32, af::dtype::f32},
                {sdnn::dtype::f64, af::dtype::f64},
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

        af::dim4 shapeToAfDim(const Shape& shape) {
            std::vector<int> dims = shape.getDimensions();
            return af::dim4(
                dims.size() > 0 ? dims[0] : 1,
                dims.size() > 1 ? dims[1] : 1,
                dims.size() > 2 ? dims[2] : 1,
                dims.size() > 3 ? dims[3] : 1
            );
        }

        std::vector<int> getArrayDimensionsAsIntVector(const af::array& array) {
            af::dim4 dims = array.dims();
            std::vector<int> dimVector;
            
            for (int i = 0; i < 4 && dims[i] > 1; ++i) {
                dimVector.push_back(dims[i]);
            }
            return dimVector;
        }

        Shape afDimToShape(const af::dim4& dim){
            std::vector<int> dims(dim.ndims());
            for (int i = 0; i < dim.ndims(); ++i) {
                dims[i] = dim[i];
            }
            return Shape(dims);
        }

        double getElementAsDouble(const af::array& arr, dim_t index) {
            return arr(index).scalar<double>();
        }
    } // namespace utils
} // namespace sdnn

#endif // UTILS_CPP