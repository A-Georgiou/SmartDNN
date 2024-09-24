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
    std::vector<bool> broadcast_dims_;

    void calculateBroadcastFactors() {
        const auto& orig_shape = original_.shape();
        broadcast_dims_.resize(broadcasted_shape_.rank(), false);
        original_strides_.resize(broadcasted_shape_.rank(), 0);
        
        size_t orig_dim = orig_shape.rank() - 1;
        for (int i = broadcasted_shape_.rank() - 1; i >= 0; --i) {
            if (orig_dim < 0 || orig_shape[orig_dim] != broadcasted_shape_[i]) {
                broadcast_dims_[i] = true;
            } else {
                original_strides_[i] = original_.shape().getStride()[orig_dim];
                --orig_dim;
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
        std::vector<size_t> indices(broadcasted_shape_.rank());
        for (int i = broadcasted_shape_.rank() - 1; i >= 0; --i) {
            indices[i] = index % broadcasted_shape_[i];
            index /= broadcasted_shape_[i];
        }

        size_t flatIndex = 0;
        for (size_t i = 0; i < broadcasted_shape_.rank(); ++i) {
            if (!broadcast_dims_[i]) {
                flatIndex += indices[i] * original_strides_[i];
            }
        }

        return original_.template at<T>(flatIndex);
    }

    const Shape& shape() const { return broadcasted_shape_; }
};

} // namespace sdnn

#endif // BROADCAST_VIEW_HPP