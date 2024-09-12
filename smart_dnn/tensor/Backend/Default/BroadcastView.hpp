#ifndef BROADCAST_VIEW_HPP
#define BROADCAST_VIEW_HPP

#include <vector>
#include <stdexcept>
#include "smart_dnn/shape/ShapeOperations.hpp"
#include "smart_dnn/shape/Shape.hpp"

namespace sdnn {

template<typename T, typename DeviceType>
class BroadcastView {
private:
    const TensorData<T, DeviceType>& original_;
    Shape broadcasted_shape_;
    std::vector<int> broadcast_factors_;

    void calculateBroadcastFactors() {
        const auto& orig_shape = original_.shape();
        broadcast_factors_.resize(broadcasted_shape_.rank(), 1);
        
        int orig_dim = orig_shape.rank() - 1;
        int broadcast_dim = broadcasted_shape_.rank() - 1;

        while (broadcast_dim >= 0) {
            int orig_size = (orig_dim >= 0) ? orig_shape[orig_dim] : 1;
            int broadcast_size = broadcasted_shape_[broadcast_dim];

            if (orig_size == broadcast_size) {
                broadcast_factors_[broadcast_dim] = 1;
            } else if (orig_size == 1) {
                broadcast_factors_[broadcast_dim] = 0;
            } else {
                throw std::runtime_error("Unexpected broadcast scenario");
            }

            --broadcast_dim;
            --orig_dim;
        }
    }

    int mapIndex(const std::vector<int>& indices) const {
        int flatIndex = 0;
        const auto& orig_shape = original_.shape();
        const auto& orig_stride = original_.shape().getStride();
        int orig_dim = orig_shape.rank() - 1;

        for (int i = broadcasted_shape_.rank() - 1; i >= 0 && orig_dim >= 0; --i) {
            if (broadcast_factors_[i] == 1) {
                flatIndex += indices[i] * orig_stride[orig_dim];
            }
            --orig_dim;
        }
        return flatIndex;
    }

public:
    BroadcastView(const TensorData<T, DeviceType>& tensor, const Shape& target_shape)
        : original_(tensor), broadcasted_shape_(target_shape) {
        if (!ShapeOperations::areBroadcastable(original_.shape(), target_shape)) {
            throw std::runtime_error("Shapes are not broadcastable! Mismatch between shapes " + original_.shape().toString() + " and " + target_shape.toString());
        }
        calculateBroadcastFactors();
    }

    const Shape& shape() const { return broadcasted_shape_; }

    T operator[](const std::vector<int>& indices) const {
        if constexpr (std::is_same_v<DeviceType, CPUDevice>) {
            return original_.data()[mapIndex(indices)];
        } else {
            throw std::runtime_error("Direct indexing is not supported for GPU tensors in BroadcastView");
        }
    }

    class Iterator {
        private:
            const BroadcastView& view_;
            std::vector<int> current_indices_;
            bool is_end_;

        public:
            // Required type aliases for MSVC
            using iterator_category = std::forward_iterator_tag;
            using value_type = T;
            using difference_type = std::ptrdiff_t;
            using pointer = T*;
            using reference = T&;

            Iterator(const BroadcastView& view, bool end = false)
                : view_(view), is_end_(end) {
                if (!end) {
                    current_indices_.resize(view.shape().rank(), 0);
                }
            }

            T operator*() const {
                if (is_end_) {
                    throw std::out_of_range("Attempting to dereference end iterator");
                }
                return view_[current_indices_];
            }

            Iterator& operator++() {
                if (is_end_) return *this;

                for (int i = current_indices_.size() - 1; i >= 0; --i) {
                    ++current_indices_[i];
                    if (current_indices_[i] < view_.shape()[i]) {
                        return *this;
                    }
                    current_indices_[i] = 0;
                }

                is_end_ = true;
                return *this;
            }

            bool operator!=(const Iterator& other) const {
                return is_end_ != other.is_end_ || (!is_end_ && current_indices_ != other.current_indices_);
            }

            bool operator==(const Iterator& other) const {
                return !(*this != other);
            }
        };

    std::string toString() const {
        std::stringstream ss;
        ss << "BroadcastView of shape " << broadcasted_shape_.toString() << " from original shape " << original_.shape().toString();
        return ss.str();
    }

    // Only provide iterators for CPU tensors
    template<typename D = DeviceType>
    typename std::enable_if<std::is_same<D, CPUDevice>::value, Iterator>::type
    begin() const { return Iterator(*this); }

    template<typename D = DeviceType>
    typename std::enable_if<std::is_same<D, CPUDevice>::value, Iterator>::type
    end() const { return Iterator(*this, true); }

};

} // namespace sdnn

#endif // BROADCAST_VIEW_HPP