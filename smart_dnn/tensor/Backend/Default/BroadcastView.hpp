#ifndef BROADCAST_VIEW_HPP
#define BROADCAST_VIEW_HPP

#include <vector>
#include <stdexcept>
#include "smart_dnn/shape/ShapeOperations.hpp"
#include "smart_dnn/shape/Shape.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/TensorIndex.hpp"

namespace sdnn {

template<typename T>
class BroadcastView {
private:
    const Tensor& original_;
    Shape broadcasted_shape_;
    std::vector<size_t> original_strides_;
    std::vector<size_t> broadcast_strides_;

    void calculateBroadcastFactors() {
        const auto& orig_shape = original_.shape();
        original_strides_.resize(broadcasted_shape_.rank(), 1);
        broadcast_strides_.resize(broadcasted_shape_.rank(), 0);
        
        size_t orig_dim = orig_shape.rank() - 1;
        size_t broadcast_stride = 1;
        for (int i = broadcasted_shape_.rank() - 1; i >= 0; --i) {
            if (orig_dim >= 0 && orig_shape[orig_dim] == broadcasted_shape_[i]) {
                original_strides_[i] = original_.shape().getStride()[orig_dim];
                broadcast_strides_[i] = broadcast_stride;
                broadcast_stride *= broadcasted_shape_[i];
                --orig_dim;
            } else if (orig_dim >= 0 && orig_shape[orig_dim] == 1) {
                original_strides_[i] = 0;
                broadcast_strides_[i] = broadcast_stride;
                broadcast_stride *= broadcasted_shape_[i];
                --orig_dim;
            } else {
                original_strides_[i] = 0; 
                broadcast_strides_[i] = broadcast_stride;
                broadcast_stride *= broadcasted_shape_[i];
            }
        }
    }

public:
    BroadcastView(const Tensor& tensor, const Shape& target_shape)
        : original_(tensor), broadcasted_shape_(target_shape) {
        if (!ShapeOperations::areBroadcastable(tensor.shape(), target_shape)) {
            throw std::runtime_error("Shapes are not broadcastable!");
        }
        calculateBroadcastFactors();
    }

    T operator[](size_t index) const {
        size_t originalIndex = 0;
        for (size_t i = 0; i < broadcasted_shape_.rank(); ++i) {
            size_t dim_index = (index / broadcast_strides_[i]) % broadcasted_shape_[i];
            originalIndex += dim_index * original_strides_[i];
        }
        return original_.template at<T>(originalIndex);
    }

    const Shape& shape() const { return broadcasted_shape_; }
};

} // namespace sdnn

#endif // BROADCAST_VIEW_HPP