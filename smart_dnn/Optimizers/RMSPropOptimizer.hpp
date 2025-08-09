#ifndef RMSPROP_OPTIMIZER_HPP
#define RMSPROP_OPTIMIZER_HPP

#include <unordered_map>
#include <cmath>
#include <memory>
#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Optimizer.hpp"

namespace smart_dnn {

template <typename T = float>
struct RMSPropOptions {
    T learningRate = T(0.01);
    T alpha = T(0.99);        // Decay rate for moving average of squared gradients
    T epsilon = T(1e-8);      // Small constant for numerical stability
    T momentum = T(0);        // Momentum factor (optional)
    bool centered = false;    // If true, use centered RMSProp
    T weightDecay = T(0);     // Weight decay (L2 regularization)
};

template <typename T = float>
class RMSPropOptimizer : public Optimizer<T> {
public:
    RMSPropOptimizer(const RMSPropOptions<T>& options = {})
        : learningRate(options.learningRate),
          alpha(options.alpha),
          epsilon(options.epsilon),
          momentum(options.momentum),
          centered(options.centered),
          weightDecay(options.weightDecay) {}

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
        auto weightsPtr = reinterpret_cast<uintptr_t>(weightData);
        
        // Get or create squared gradients buffer
        if (squaredGrads.find(weightsPtr) == squaredGrads.end()) {
            squaredGrads[weightsPtr] = std::make_shared<Tensor<T>>(weight.getShape(), T(0));
        }
        auto& sqGrads = *squaredGrads[weightsPtr];
        T* sqGradsData = sqGrads.getData().data();
        
        // For centered RMSProp, also track the mean of gradients
        std::shared_ptr<Tensor<T>> meanGradsPtr = nullptr;
        T* meanGradsData = nullptr;
        if (centered) {
            if (meanGrads.find(weightsPtr) == meanGrads.end()) {
                meanGrads[weightsPtr] = std::make_shared<Tensor<T>>(weight.getShape(), T(0));
            }
            meanGradsPtr = meanGrads[weightsPtr];
            meanGradsData = meanGradsPtr->getData().data();
        }
        
        // For momentum, track velocity buffer
        std::shared_ptr<Tensor<T>> velocityPtr = nullptr;
        T* velocityData = nullptr;
        if (momentum > T(0)) {
            if (velocityBuffers.find(weightsPtr) == velocityBuffers.end()) {
                velocityBuffers[weightsPtr] = std::make_shared<Tensor<T>>(weight.getShape(), T(0));
            }
            velocityPtr = velocityBuffers[weightsPtr];
            velocityData = velocityPtr->getData().data();
        }
        
        for (size_t j = 0; j < size; ++j) {
            T grad = gradData[j];
            
            // Add weight decay if specified
            if (weightDecay != T(0)) {
                grad += weightDecay * weightData[j];
            }
            
            // Update squared gradients: sq_avg = alpha * sq_avg + (1 - alpha) * grad^2
            sqGradsData[j] = alpha * sqGradsData[j] + (T(1) - alpha) * grad * grad;
            
            T denom;
            if (centered) {
                // Update mean gradients: mean_avg = alpha * mean_avg + (1 - alpha) * grad
                meanGradsData[j] = alpha * meanGradsData[j] + (T(1) - alpha) * grad;
                
                // Centered RMSProp: denominator = sqrt(sq_avg - mean_avg^2 + epsilon)
                T variance = sqGradsData[j] - meanGradsData[j] * meanGradsData[j];
                denom = std::sqrt(variance + epsilon);
            } else {
                // Standard RMSProp: denominator = sqrt(sq_avg + epsilon)
                denom = std::sqrt(sqGradsData[j] + epsilon);
            }
            
            // Compute update
            T update = grad / denom;
            
            if (momentum > T(0)) {
                // With momentum: v = momentum * v + update, weight = weight - lr * v
                velocityData[j] = momentum * velocityData[j] + update;
                weightMutableData[j] -= currentLR * velocityData[j];
            } else {
                // Without momentum: weight = weight - lr * update
                weightMutableData[j] -= currentLR * update;
            }
        }
    }

    T learningRate;
    T alpha;
    T epsilon;
    T momentum;
    bool centered;
    T weightDecay;
    
    // Store state buffers for each tensor (identified by memory address)
    std::unordered_map<uintptr_t, std::shared_ptr<Tensor<T>>> squaredGrads;
    std::unordered_map<uintptr_t, std::shared_ptr<Tensor<T>>> meanGrads;      // For centered RMSProp
    std::unordered_map<uintptr_t, std::shared_ptr<Tensor<T>>> velocityBuffers; // For momentum
};

} // namespace smart_dnn

#endif // RMSPROP_OPTIMIZER_HPP