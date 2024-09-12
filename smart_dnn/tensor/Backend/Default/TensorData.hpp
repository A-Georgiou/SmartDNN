#ifndef TENSOR_DATA_HPP
#define TENSOR_DATA_HPP

#include <iomanip>
#include <functional>
#include <arrayfire.h>
#include <memory>
#include <vector>
#include <initializer_list>
#include "smart_dnn/shape/Shape.hpp"
#include "smart_dnn/shape/ShapeOperations.hpp"
#include "smart_dnn/tensor/TensorAdapterBase.hpp"
#include "DeviceTypes.hpp"

namespace sdnn {

// Specialization for CPUDevice
class CPUTensor : TensorAdapter {
public:
    

private:
    Shape shape_;
    std::unique_ptr<dtype[]> data_;
};

} // namespace sdnn

#include "TensorDataCPU.impl.hpp"

#endif // TENSOR_DATA_HPP
