#ifndef SMART_DNN_HPP
#define SMART_DNN_HPP

#include "smart_dnn/Layer.hpp"
#include "smart_dnn/Loss.hpp"
#include "smart_dnn/Optimizer.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"
#include <vector>

namespace smart_dnn {

template <typename T=float>
class SmartDNN {
public:
    SmartDNN() = default;
    ~SmartDNN() = default;

    template<typename LayerType>
    void addLayer(LayerType&& layer) {
        layers.push_back(std::make_unique<std::decay_t<LayerType>>(std::forward<LayerType>(layer)));
    }

    template<typename LossType, typename OptimizerType>
    void compile(LossType&& loss, OptimizerType&& optimizer) {
        lossFunction = std::make_unique<std::decay_t<LossType>>(std::forward<LossType>(loss));
        this->optimizer = std::make_unique<std::decay_t<OptimizerType>>(std::forward<OptimizerType>(optimizer));
    }

    void train(const std::vector<Tensor<T>>& inputs, const std::vector<Tensor<T>>& targets, int epochs);
    Tensor<T> predict(const Tensor<T>& input);
    std::vector<Tensor<T>> predict(const std::vector<Tensor<T>>& inputs);

    Layer<T>* getLayer(size_t index) const;

    void backward(const Tensor<T>& gradOutput);
    void updateWeights();

    void trainingMode();
    void evalMode();

    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);

private:
    void setTrainingMode(bool trainingMode);

    std::vector<std::unique_ptr<Layer<T>>> layers;
    std::unique_ptr<Loss<T>> lossFunction;
    std::unique_ptr<Optimizer<T>> optimizer;
};


} // namespace smart_dnn

#include "SmartDNN/SmartDNN.impl.hpp"

#endif // SMART_DNN_HPP