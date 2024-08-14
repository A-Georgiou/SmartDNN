#include <Layer.hpp>
#include <Loss.hpp>
#include <Optimizer.hpp>
#include <Tensor.hpp>
#include <vector>

class SmartDNN {
public:
    SmartDNN();
    ~SmartDNN();

    void addLayer(Layer* layer);
    void compile(Loss* loss, Optimizer* optimizer);
    void train(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, int epochs, float learningRate);
    Tensor predict(const Tensor& input);

    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);

private:
    std::vector<Layer*> layers;
    Loss* lossFunction;
    Optimizer* optimizer;
};