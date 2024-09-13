#include "smart_dnn/tensor/Backend/Default/CPUTensor.hpp"

namespace sdnn{

class TensorView {
    public:
        TensorView(CPUTensor& tensor, size_t index)
            : tensor_(tensor), index_(index) {}

        // Implicit conversion to double for reading values
        operator double() const {
            return tensor_.getValueAsDouble(index_);
        }

        // Assignment operator for setting values
        TensorView& operator=(double value) {
            tensor_.setValueFromDouble(index_, value);
            return *this;
        }

    private:
        CPUTensor& tensor_;
        size_t index_;
};

}