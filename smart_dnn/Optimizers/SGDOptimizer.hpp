#ifndef SGD_OPTIMIZER_HPP
#define SGD_OPTIMIZER_HPP

#include <unordered_map>
#include <cmath>
#include <memory>
#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Optimizer.hpp"

namespace smart_dnn {

template <typename T = float>
struct SGDOptions {
    T learningRate = T(0.01);
    T momentum = T(0.9);
    T dampening = T(0);
    T weightDecay = T(0);
    bool nesterov = false;
};

template <typename T = float>
class SGDOptimizer : public Optimizer<T> {
public:
    SGDOptimizer(const SGDOptions<T>& options = {})
        : learningRate(options.learningRate),
          momentum(options.momentum),
          dampening(options.dampening),
          weightDecay(options.weightDecay),
          nesterov(options.nesterov) {}

    void optimize(const std::vector<std::reference_wrapper<Tensor<T>>>& weights,
                  const std::vector<std::reference_wrapper<Tensor<T>>>& gradients,
                  T learningRateOverride = T(-1)) override {
        if (weights.size() != gradients.size()) {
            throw std::invalid_argument("Weights and gradients size mismatch!");
        }

        T currentLR = (learningRateOverride > T(0)) ? learningRateOverride : learningRate;

        for (size_t i = 0; i < weights.size(); ++i) {
            updateTensor(weights[i].get(), gradients[i].get(), currentLR);
        }
    }

private:
    void updateTensor(Tensor<T>& weight, const Tensor<T>& gradient, T currentLR) {
        const T* weightData = weight.getData().data();
        const T* gradData = gradient.getData().data();
        T* weightMutableData = weight.getData().data();
        
        size_t size = weight.getShape().size();
        
        // Get or create momentum buffer for this tensor
        auto weightsPtr = reinterpret_cast<uintptr_t>(weightData);
        
        // Check if momentum buffer exists, create if it doesn't
        if (momentumBuffers.find(weightsPtr) == momentumBuffers.end()) {
            momentumBuffers[weightsPtr] = std::make_shared<Tensor<T>>(weight.getShape(), T(0));
        }
        
        auto& momentumBuffer = *momentumBuffers[weightsPtr];
        
        T* momentumData = momentumBuffer.getData().data();
        
        for (size_t j = 0; j < size; ++j) {
            T grad = gradData[j];
            
            // Add weight decay if specified
            if (weightDecay != T(0)) {
                grad += weightDecay * weightData[j];
            }
            
            if (momentum != T(0)) {
                // Update momentum: v = momentum * v + (1 - dampening) * grad
                momentumData[j] = momentum * momentumData[j] + (T(1) - dampening) * grad;
                
                if (nesterov) {
                    // Nesterov momentum: grad = grad + momentum * v
                    grad = grad + momentum * momentumData[j];
                } else {
                    // Standard momentum: grad = v
                    grad = momentumData[j];
                }
            }
            
            // Update weight: w = w - lr * grad
            weightMutableData[j] -= currentLR * grad;
        }
    }

    T learningRate;
    T momentum;
    T dampening;
    T weightDecay;
    bool nesterov;
    
    // Store momentum buffers for each tensor (identified by memory address)
    std::unordered_map<uintptr_t, std::shared_ptr<Tensor<T>>> momentumBuffers;
};

} // namespace smart_dnn

#endif // SGD_OPTIMIZER_HPP