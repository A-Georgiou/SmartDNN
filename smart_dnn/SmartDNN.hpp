#ifndef SMART_DNN_HPP
#define SMART_DNN_HPP

#include "Layer.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include "tensor/Tensor.hpp"
#include <vector>

namespace sdnn {

template <typename T=float>
class SmartDNN {
public:
    SmartDNN();
    ~SmartDNN();

    void addLayer(Layer<T>* layer);
    void compile(Loss<T>* loss, Optimizer<T>* optimizer);
    void train(const std::vector<Tensor<T>>& inputs, const std::vector<Tensor<T>>& targets, int epochs);
    Tensor<T> predict(const Tensor<T>& input);
    std::vector<Tensor<T>> predict(const std::vector<Tensor<T>>& inputs);

    Layer<T>* getLayer(size_t index) const;

    void backward(const Tensor<T>& gradOutput);
    void updateWeights(Optimizer<T>& optimizer);

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

} // namespace sdnn

#include "model/SmartDNN.impl.hpp"

#endif // SMART_DNN_HPP