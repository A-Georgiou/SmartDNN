#ifndef LAYER_HPP
#define LAYER_HPP

#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Tensor/TensorIO.hpp"
#include "smart_dnn/Optimizer.hpp"
#include <functional>
#include <stdexcept>
#include <vector>

namespace smart_dnn {

    // A learned tensor and its gradient from the most recent backward pass.
    template <typename T>
    struct Parameter {
        std::reference_wrapper<Tensor<T>> value;
        std::reference_wrapper<Tensor<T>> gradient;

        bool hasGradient() const { return gradient.get().getShape() == value.get().getShape(); }
    };

    // Stands in for a gradient until the first backward pass, so inference allocates no gradient buffers.
    template <typename T>
    Tensor<T> noGradient() {
        return Tensor<T>(Shape({1}), T(0));
    }

    template <typename T>
    class Layer {
    public:
        virtual ~Layer() = default;

        virtual Tensor<T> forward(const Tensor<T>& input) = 0;
        virtual Tensor<T> backward(const Tensor<T>& gradOutput) = 0;

        // Learned tensors with their gradients, in a fixed order; layers without weights return none.
        virtual std::vector<Parameter<T>> parameters() { return {}; }

        virtual inline void updateWeights(Optimizer<T>& optimizer) {
            std::vector<std::reference_wrapper<Tensor<T>>> values;
            std::vector<std::reference_wrapper<Tensor<T>>> gradients;
            for (const Parameter<T>& parameter : parameters()) {
                if (!parameter.hasGradient()) {
                    throw std::logic_error("updateWeights needs a backward pass first");
                }
                values.push_back(parameter.value);
                gradients.push_back(parameter.gradient);
            }
            if (!values.empty()) {
                optimizer.optimize(values, gradients);
            }
        }
        virtual inline void setTrainingMode(bool mode) { trainingMode = mode; }

        // Learned parameters for SmartDNN::saveModel and loadModel; layers without parameters write nothing.
        virtual inline void save(std::ostream& os) const { (void)os; }
        virtual inline void load(std::istream& is) { (void)is; }

    protected:
        bool trainingMode = true;
    };

} // namespace smart_dnn

#endif // LAYER_HPP
