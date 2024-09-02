#ifndef LAYER_HPP
#define LAYER_HPP

#include "Tensor/Tensor.hpp"
#include "Optimizer.hpp"

namespace smart_dnn {

    class Layer {
    public:
        virtual ~Layer() = default;

        virtual void setTrainingMode(bool mode) {
            trainingMode = mode;
        }

        virtual Tensor forward(const Tensor& input) = 0;
        virtual Tensor backward(const Tensor& gradOutput) = 0;
        virtual void updateWeights(Optimizer& optimizer) = 0;

    protected:
        bool trainingMode = true;
    };

} // namespace smart_dnn

#endif // LAYER_HPP
