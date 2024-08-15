#include "Tensor.hpp"

class Loss {
public:
    virtual ~Loss() = default;
    virtual float compute(const Tensor& prediction, const Tensor& target) = 0;
    virtual Tensor gradient(const Tensor& prediction, const Tensor& target) = 0;
    
    virtual void save(std::ostream& os) const = 0;
    virtual void load(std::istream& is) = 0;
};
