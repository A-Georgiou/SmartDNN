#ifndef LOSS_HPP
#define LOSS_HPP

#include "smart_dnn/tensor/TensorBase.hpp"

namespace sdnn {

class Loss {
public:
    virtual ~Loss() = default;

    virtual Tensor compute(const Tensor& prediction, const Tensor& target) = 0;
    virtual Tensor gradient(const Tensor& prediction, const Tensor& target) = 0;
    
    virtual inline void save(std::ostream& os) const {
        (void)os;
    };
    virtual inline void load(std::istream& is) {
        (void)is;
    };
};

} // namespace sdnn

#endif // LOSS_HPP