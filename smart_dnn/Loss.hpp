#ifndef LOSS_HPP
#define LOSS_HPP

#include "smart_dnn/Tensor/Tensor.hpp"

namespace smart_dnn {

template <typename T>
class Loss {
public:
    virtual ~Loss() = default;

    virtual Tensor<T> compute(const Tensor<T>& prediction, const Tensor<T>& target) = 0;
    virtual Tensor<T> gradient(const Tensor<T>& prediction, const Tensor<T>& target) = 0;
    
    virtual inline void save(std::ostream& os) const {
        (void)os;
    };
    virtual inline void load(std::istream& is) {
        (void)is;
    };
};

} // namespace smart_dnn

#endif // LOSS_HPP