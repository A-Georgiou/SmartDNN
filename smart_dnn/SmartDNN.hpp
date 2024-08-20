#ifndef SMART_DNN_HPP
#define SMART_DNN_HPP

#include "Layer.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include "Tensor.hpp"
#include <vector>

class SmartDNN {
public:
    SmartDNN();
    ~SmartDNN();

    void addLayer(Layer* layer);
    void compile(Loss* loss, Optimizer* optimizer);
    void train(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, int epochs, float learningRate);
    Tensor predict(const Tensor& input);
    std::vector<Tensor> predict(const std::vector<Tensor>& inputs);

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

#endif // SMART_DNN_HPP