#ifndef LAYER_HPP
#define LAYER_HPP

#include "Tensor/Tensor.hpp"
#include "Optimizer.hpp"
#include "Tensor/TensorConfig.hpp"

namespace smart_dnn {

    class Layer {
    public:
        virtual ~Layer() = default;

        virtual void setTrainingMode(bool mode) {
            trainingMode = mode;
        }

        virtual ConfiguredTensor<> forward(const ConfiguredTensor<>& input) = 0;
        virtual ConfiguredTensor<> backward(const ConfiguredTensor<>& gradOutput) = 0;
        virtual void updateWeights(Optimizer& optimizer) = 0;

    protected:
        bool trainingMode = true;
    };

} // namespace smart_dnn

#endif // LAYER_HPP
