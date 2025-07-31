#ifndef SLICE_VIEW_HPP
#define SLICE_VIEW_HPP

#include "smart_dnn/Shape/Shape.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Tensor/DeviceTypes.hpp"
#include <vector>
#include <stdexcept>
#include <optional>

namespace smart_dnn {

template<typename T = float, typename DeviceType = CPUDevice>
class SliceView {
private:
    const TensorData<T, DeviceType>& original_;
    std::optional<Shape> sliced_shape_;
    std::vector<int> start_indices_;
    std::vector<int> step_sizes_;

    void calculateSlicedShape() {
        std::vector<int> new_dims;
        for (size_t i = 0; i < original_.shape().rank(); ++i) {
            int dim_size = (original_.shape()[i] - start_indices_[i] + step_sizes_[i] - 1) / step_sizes_[i];
            new_dims.push_back(dim_size);
        }
        sliced_shape_ = Shape(new_dims);
    }

    int mapIndex(const std::vector<int>& indices) const {
        int flatIndex = 0;
        const auto& orig_stride = original_.shape().getStride();
        for (size_t i = 0; i < indices.size(); ++i) {
            flatIndex += (start_indices_[i] + indices[i] * step_sizes_[i]) * orig_stride[i];
        }
        return flatIndex;
    }

public:
    SliceView(const TensorData<T, DeviceType>& tensor,
              const std::vector<std::pair<int, int>>& slices,
              const std::vector<int>& steps)
        : original_(tensor) {
        if (slices.size() != static_cast<size_t>(tensor.shape().rank()) || steps.size() != static_cast<size_t>(tensor.shape().rank())) {
            throw std::runtime_error("Slice dimensions must match tensor rank");
        }

        start_indices_.reserve(slices.size());
        step_sizes_ = steps;

        for (size_t i = 0; i < slices.size(); ++i) {
            int start = slices[i].first;
            int end = slices[i].second;

            if (start < 0) start += tensor.shape()[i];
            if (end < 0) end += tensor.shape()[i];
            if (end > tensor.shape()[i]) end = tensor.shape()[i];

            start = std::max(0, std::min(start, tensor.shape()[i] - 1));
            end = std::max(start, std::min(end, tensor.shape()[i]));

            start_indices_.push_back(start);
        }

        calculateSlicedShape();
    }

    const Shape& shape() const { return *sliced_shape_; }

    T operator[](const std::vector<int>& indices) const {
        if constexpr (std::is_same_v<DeviceType, CPUDevice>) {
            return original_.data()[mapIndex(indices)];
        } else {
            throw std::runtime_error("Direct indexing is not supported for GPU tensors in SliceView");
        }
    }

    void set(const std::vector<int>& indices, T value) const {
        if constexpr (std::is_same_v<DeviceType, CPUDevice>) {
            const_cast<T*>(original_.data())[mapIndex(indices)] = value;
        } else {
            throw std::runtime_error("Direct indexing is not supported for GPU tensors in SliceView");
        }
    }

    class Iterator {
    private:
        const SliceView& view_;
        std::vector<int> current_indices_;
        bool is_end_;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        Iterator(const SliceView& view, bool end = false)
            : view_(view), is_end_(end) {
            if (!end) {
                current_indices_.resize(view.shape().rank(), 0);
            }
        }
        
        const std::vector<int>& getCurrentIndices() const {
            return current_indices_;
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
        ss << "SliceView of shape " << (*sliced_shape_).toString() << " from original shape " << original_.shape().toString();
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

} // namespace smart_dnn

#endif // SLICE_VIEW_HPP