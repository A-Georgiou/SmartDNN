#ifndef SMART_DNN_HPP
#define SMART_DNN_HPP

#include "smart_dnn/Layer.hpp"
#include "smart_dnn/Loss.hpp"
#include "smart_dnn/Optimizer.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include <vector>

namespace sdnn {

class SmartDNN {
public:
    SmartDNN();
    ~SmartDNN();

    void addLayer(Layer* layer);
    void compile(Loss* loss, Optimizer* optimizer);
    void train(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, int epochs);
    Tensor predict(const Tensor& input);
    std::vector<Tensor> predict(const std::vector<Tensor>& inputs);

    Layer* getLayer(size_t index) const;

    void backward(const Tensor& gradOutput);
    void updateWeights(Optimizer& optimizer);

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

} // namespace sdnn

#include "model/SmartDNN.impl.hpp"

#endif // SMART_DNN_HPP