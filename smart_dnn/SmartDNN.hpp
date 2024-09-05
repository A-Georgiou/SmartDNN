#ifndef SMART_DNN_HPP
#define SMART_DNN_HPP

#include "Layer.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include "Tensor/Tensor.hpp"
#include <vector>

namespace smart_dnn {

template <typename T=float>
class SmartDNN {
public:
    SmartDNN();
    ~SmartDNN();

    void addLayer(Layer<T>* layer);
    void compile(Loss<T>* loss, Optimizer<T>* optimizer);
    void train(const std::vector<Tensor<T>>& inputs, const std::vector<Tensor<T>>& targets, int epochs, float learningRate);
    Tensor<T> predict(const Tensor<T>& input);
    std::vector<Tensor<T>> predict(const std::vector<Tensor<T>>& inputs);

    Layer<T>* getLayer(size_t index) const;

    // Backward pass - compute gradients
    void backward(const Tensor<T>& gradOutput) {
        Tensor<T> grad = gradOutput;
        // Iterate over layers in reverse order
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = (*it)->backward(grad);  // Assuming each layer has its own backward method
        }
    }

    // Update weights using the optimizer
    void updateWeights(Optimizer<T>& optimizer) {
        for (auto& layer : layers) {
            layer->updateWeights(optimizer);  // Assuming each layer has its own updateWeights method
        }
    }

    void trainingMode();
    void evalMode();

    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);

private:
    void setTrainingMode(bool trainingMode);

    std::vector<Layer<T>*> layers;
    Loss<T>* lossFunction;
    Optimizer<T>* optimizer;
};

} // namespace smart_dnn

#include "SmartDNN/SmartDNN.impl.hpp"

#endif // SMART_DNN_HPP