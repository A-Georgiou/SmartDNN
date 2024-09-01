#ifndef BROADCAST_VIEW_HPP
#define BROADCAST_VIEW_HPP

#include "../Shape/ShapeOperations.hpp"
#include "../Shape/Shape.hpp"
#include "Tensor.hpp"
#include "DeviceTypes.hpp"

namespace smart_dnn {

template<typename T, typename DeviceType>
class BroadcastView {
private:
    const TensorData<T, DeviceType>& original_;
    Shape broadcasted_shape_;

    size_t mapIndex(const std::vector<int>& indices) const {
        size_t flatIndex = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            int originalDim = i < original_.shape().size() ? original_.shape()[i] : 1;
            flatIndex += (indices[i] % originalDim) * (i < original_.shape().size() ? original_.stride()[i] : 0);
        }
        return flatIndex;
    }

public:
    BroadcastView(const TensorData<T, DeviceType>& tensor, const Shape& target_shape)
        : original_(tensor), broadcasted_shape_(target_shape) {}

    const Shape& shape() const { return broadcasted_shape_; }

    T operator[](const std::vector<int>& indices) const {
        if constexpr (std::is_same_v<DeviceType, CPUDevice>) {
            return original_.data()[mapIndex(indices)];
        } else {
            // For GPUDevice, this operation is not supported
            throw std::runtime_error("Direct indexing is not supported for GPU tensors in BroadcastView");
        }
    }

    // Iterator for the broadcasted view (only for CPU tensors)
    class Iterator {
    private:
        const BroadcastView& view_;
        std::vector<int> current_indices_;

    public:
        Iterator(const BroadcastView& view, bool end = false)
            : view_(view), current_indices_(view.shape().size(), 0) {
            if (end) {
                current_indices_[0] = view.shape()[0];
            }
        }

        T operator*() const {
            return view_[current_indices_];
        }

        Iterator& operator++() {
            for (int i = current_indices_.size() - 1; i >= 0; --i) {
                if (++current_indices_[i] < view_.shape()[i]) {
                    break;
                }
                current_indices_[i] = 0;
            }
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            return current_indices_ != other.current_indices_;
        }
    };

    // Only provide iterators for CPU tensors
    template<typename D = DeviceType>
    typename std::enable_if<std::is_same<D, CPUDevice>::value, Iterator>::type
    begin() const { return Iterator(*this); }

    template<typename D = DeviceType>
    typename std::enable_if<std::is_same<D, CPUDevice>::value, Iterator>::type
    end() const { return Iterator(*this, true); }
};

} // namespace smart_dnn

#endif // BROADCAST_VIEW_HPP