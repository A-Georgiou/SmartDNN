#ifndef LAYER_HPP
#define LAYER_HPP

#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Optimizer.hpp"

namespace smart_dnn {

    template <typename T>
    class Layer {
    public:
        virtual ~Layer() = default;

        virtual Tensor<T> forward(const Tensor<T>& input) = 0;
        virtual Tensor<T> backward(const Tensor<T>& gradOutput) = 0;

        virtual inline void updateWeights(Optimizer<T>& optimizer) { (void)optimizer; }
        virtual inline void setTrainingMode(bool mode) { trainingMode = mode; }

    protected:
        bool trainingMode = true;
    };

} // namespace smart_dnn

#endif // LAYER_HPP
