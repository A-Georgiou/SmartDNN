#ifndef UTILS_HPP
#define UTILS_HPP


#include <cmath>
#include "smart_dnn/DTypes.hpp"
#include "smart_dnn/shape/Shape.hpp"
#include "arrayfire.h"
#include <af/data.h>

/*

    Set of helper utilities for ArrayFire Tensor implementation

*/

namespace sdnn {
    namespace utils {
    
        af::dtype sdnnToAfType(const sdnn::dtype type);
        sdnn::dtype afToSdnnType(const af::dtype type);
        af::dim4 shapeToAfDim(const Shape& shape);
        std::vector<int> getArrayDimensionsAsIntVector(const af::array& array);
        Shape afDimToShape(const af::dim4& dim);
        double getElementAsDouble(const af::array& arr, dim_t index, sdnn::dtype type);

    } // namespace utils
} // namespace sdnn

#endif // UTILS_HPP