#ifndef TENSOR_VIEW_HPP
#define TENSOR_VIEW_HPP

#include "smart_dnn/tensor/TensorAdapterBase.hpp"

namespace sdnn{

class TensorView {
    public:
        TensorView(TensorAdapter& tensor, size_t index)
            : tensor_(tensor), index_(index) {}

        operator double() const {
            return tensor_.getValueAsDouble(index_);
        }

        TensorView& operator=(double value) {
            tensor_.setValueFromDouble(index_, value);
            return *this;
        }

    private:
        TensorAdapter& tensor_;
        size_t index_;
};

}

#endif // TENSOR_VIEW_HPP