#include <Tensor.hpp>

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& gradOutput) = 0;
    virtual void updateWeights(float learningRate) = 0;
    
    virtual void save(std::ostream& os) const = 0;
    virtual void load(std::istream& is) = 0;
};