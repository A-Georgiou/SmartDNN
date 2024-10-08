#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "smart_dnn/Activation.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

namespace sdnn {

/*

    Softmax Activation Function
    ---------------------------
    
    f(x) = exp(x) / sum(exp(x))
    f'(x) = f(x) * (1 - f(x))

*/
class Softmax : public Activation {
public:
    Tensor forward(const Tensor& input) const override {
        if (input.shape().rank() < 2) {
            throw std::invalid_argument("Input must have at least 2 dimensions (batch_size, features)");
        }

        std::vector<size_t> featureDims(input.shape().rank() - 1);
        std::iota(featureDims.begin(), featureDims.end(), 1);  // [1, 2, ..., rank-1]

        Tensor maxValues = max(input, featureDims, true);
        Tensor shiftedInput = input - maxValues;
        Tensor expInput = exp(shiftedInput);

        Tensor sumExp = sum(expInput, featureDims, true);
                
        return expInput / sumExp;
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        if (input.shape().rank() < 2 || gradOutput.shape().rank() < 2) {
            throw std::invalid_argument("Input and gradOutput must have at least 2 dimensions (batch_size, features)");
        }

        Tensor softmaxOutput = forward(input);
        
        std::vector<size_t> featureDims(input.shape().rank() - 1);
        std::iota(featureDims.begin(), featureDims.end(), 1);  // [1, 2, ..., rank-1]

        Tensor sumGradOutput = sum(gradOutput * softmaxOutput, featureDims, true);
        return softmaxOutput * (gradOutput - sumGradOutput);
    }
};

} // namespace sdnn

#endif // SOFTMAX_HPP