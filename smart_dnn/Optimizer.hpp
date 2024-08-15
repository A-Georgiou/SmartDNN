#include <vector>
#include "Tensor.hpp"

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void optimize(std::vector<Tensor>& weights, std::vector<Tensor>& gradients, float learningRate) = 0;
    
    virtual void save(std::ostream& os) const = 0;
    virtual void load(std::istream& is) = 0;
};

