#ifndef SMART_DNN_HPP
#define SMART_DNN_HPP

#include "Layer.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include "Tensor/Tensor.hpp"
#include "Tensor/TensorConfig.hpp"
#include <vector>

namespace smart_dnn {

class SmartDNN {
public:
    SmartDNN();
    ~SmartDNN();

    void addLayer(Layer* layer);
    void compile(Loss* loss, Optimizer* optimizer);
    void train(const std::vector<ConfiguredTensor<>>& inputs, const std::vector<ConfiguredTensor<>>& targets, int epochs, float learningRate);
    ConfiguredTensor<> predict(const ConfiguredTensor<>& input);
    std::vector<ConfiguredTensor<>> predict(const std::vector<ConfiguredTensor<>>& inputs);

    void trainingMode();
    void evalMode();

    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);

private:
    void setTrainingMode(bool trainingMode);

    std::vector<Layer*> layers;
    Loss* lossFunction;
    Optimizer* optimizer;
};

} // namespace smart_dnn

#include "SmartDNN/SmartDNN.impl.hpp"

#endif // SMART_DNN_HPP