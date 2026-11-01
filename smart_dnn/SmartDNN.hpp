#ifndef SMART_DNN_HPP
#define SMART_DNN_HPP

#include "smart_dnn/Layer.hpp"
#include "smart_dnn/Loss.hpp"
#include "smart_dnn/Optimizer.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>

namespace smart_dnn {

// How saveModel stores weights: the model's own floating-point type, or 8-bit blocks about a quarter of the size.
enum class WeightFormat { Float, Int8 };

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
    // One forward, backward and optimizer step on a batch, returning its loss; a positive learningRate overrides the optimizer's.
    T trainStep(const Tensor<T>& input, const Tensor<T>& target, T learningRate = T(-1));
    Tensor<T> predict(const Tensor<T>& input);
    std::vector<Tensor<T>> predict(const std::vector<Tensor<T>>& inputs);

    Layer<T>* getLayer(size_t index) const;

    void backward(const Tensor<T>& gradOutput);
    void updateWeights(T learningRate = T(-1));

    // Scales all gradients together when their combined L2 norm exceeds maxNorm; 0 turns clipping off.
    void setMaxGradientNorm(T maxNorm);

    void trainingMode();
    void evalMode();

    void saveModel(const std::string& filename, WeightFormat format = WeightFormat::Float) const;
    void loadModel(const std::string& filename);

private:
    // Model files start with "SDNN" (little-endian), a format version, the bytes stored per weight (sizeof(T), or 1 for Int8) and the layer count.
    static constexpr uint32_t modelMagic = 0x4E4E4453;
    static constexpr uint32_t modelVersion = 1;

    void setTrainingMode(bool trainingMode);
    void clipGradients(const std::vector<std::reference_wrapper<Tensor<T>>>& gradients) const;

    std::vector<std::unique_ptr<Layer<T>>> layers;
    std::unique_ptr<Loss<T>> lossFunction;
    std::unique_ptr<Optimizer<T>> optimizer;
    T maxGradientNorm = T(0);
};


} // namespace smart_dnn

#include "SmartDNN/SmartDNN.impl.hpp"

#endif // SMART_DNN_HPP