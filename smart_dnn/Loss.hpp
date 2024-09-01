#ifndef LOSS_HPP
#define LOSS_HPP

#include "Tensor/Tensor.hpp"
#include "Tensor/TensorConfig.hpp"

namespace smart_dnn {

class Loss {
public:
    virtual ~Loss() = default;
    virtual float compute(const ConfiguredTensor<>& prediction, const ConfiguredTensor<>& target) = 0;
    virtual ConfiguredTensor<> gradient(const ConfiguredTensor<>& prediction, const ConfiguredTensor<>& target) = 0;
    
    virtual void save(std::ostream& os) const = 0;
    virtual void load(std::istream& is) = 0;
};

} // namespace smart_dnn

#endif // LOSS_HPP