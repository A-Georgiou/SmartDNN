#ifndef LAYER_HPP
#define LAYER_HPP

#include "Tensor/Tensor.hpp"
#include "Optimizer.hpp"

namespace smart_dnn {

    template <typename T>
    class Layer {
    public:
        virtual ~Layer() = default;

        virtual void setTrainingMode(bool mode) { trainingMode = mode; }

        virtual Tensor<T> forward(const Tensor<T>& input) = 0;
        virtual Tensor<T> backward(const Tensor<T>& gradOutput) = 0;
        virtual void updateWeights(Optimizer<T>& optimizer) = 0;

    protected:
        bool trainingMode = true;
    };

} // namespace smart_dnn

#endif // LAYER_HPP
