#ifndef LAYER_HPP
#define LAYER_HPP

#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/Optimizer.hpp"

namespace sdnn {

    class Layer {
    public:
        virtual ~Layer() = default;

        virtual Tensor forward(const Tensor& input) = 0;
        virtual Tensor backward(const Tensor& gradOutput) = 0;

        virtual inline void updateWeights(Optimizer& optimizer) { (void)optimizer; }
        virtual inline void setTrainingMode(bool mode) { trainingMode = mode; }

    protected:
        bool trainingMode = true;
    };

} // namespace sdnn

#endif // LAYER_HPP
