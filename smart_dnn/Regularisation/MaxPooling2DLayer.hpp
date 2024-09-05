#ifndef MAX_POOLING_2D_LAYER_HPP
#define MAX_POOLING_2D_LAYER_HPP

#include <algorithm>
#include <limits>
#include "../Layer.hpp"
#include "../Tensor/Tensor.hpp"

namespace smart_dnn {

template <typename T=float>
class MaxPooling2DLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    MaxPooling2DLayer(int poolSize) : MaxPooling2DLayer(poolSize, poolSize, 0) {}
    MaxPooling2DLayer(int poolSize, int stride) : MaxPooling2DLayer(poolSize, stride, 0) {}
    MaxPooling2DLayer(int poolSize, int stride, int padding) : poolSize(poolSize), stride(stride), padding(padding) {}

    TensorType forward(const TensorType& input) override {
        if (input.getShape().rank() != 4) {
            throw std::runtime_error("MaxPooling2DLayer: input tensor must have rank 4");
        }

        this->input = input;

        int batchSize = input.getShape()[0];
        int inputChannels = input.getShape()[1];  // Number of channels should remain unchanged
        int inputHeight = input.getShape()[2];
        int inputWidth = input.getShape()[3];

        // Calculate output dimensions based on stride and pooling size
        int outputHeight = (inputHeight - poolSize) / stride + 1;
        int outputWidth = (inputWidth - poolSize) / stride + 1;

        // Output shape retains the number of channels
        TensorType output({batchSize, inputChannels, outputHeight, outputWidth});

        for (int n = 0; n < batchSize; ++n) {
            for (int ic = 0; ic < inputChannels; ++ic) {  // Ensure channel is retained
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        auto [maxVal, maxIh, maxIw] = findMaxInPoolWindow(input, n, ic, oh, ow);
                        output.at({n, ic, oh, ow}) = maxVal;
                    }
                }
            }
        }
        return output;
    }

    TensorType backward(const TensorType& gradOutput) override {
        if (input.has_value() == false) {
            throw std::runtime_error("MaxPooling2DLayer: input tensor is not set");
        }

        TensorType& tensorValue = *input;
        TensorType gradInput(tensorValue.getShape(), 0.0f);

        int batchSize = tensorValue.getShape()[0];
        int inputChannels = tensorValue.getShape()[1];
        int outputHeight = gradOutput.getShape()[2];
        int outputWidth = gradOutput.getShape()[3];

        for (int n = 0; n < batchSize; ++n) {
            for (int ic = 0; ic < inputChannels; ++ic) {
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        auto [maxVal, maxIh, maxIw] = findMaxInPoolWindow(tensorValue, n, ic, oh, ow);
                        gradInput.at({n, ic, maxIh, maxIw}) += gradOutput.at({n, ic, oh, ow});
                    }
                }
            }
        }
        return gradInput;
    }

private:
    int poolSize;
    int stride;
    int padding;
    std::optional<TensorType> input;

    std::tuple<T, int, int> findMaxInPoolWindow(const TensorType& tensor, int n, int ic, int oh, int ow) {
        T maxVal = -std::numeric_limits<T>::infinity();
        int maxIh = 0, maxIw = 0;
        int ihStart = oh * stride - padding;
        int iwStart = ow * stride - padding;

        for (int ph = 0; ph < poolSize; ++ph) {
            int ih = ihStart + ph;
            if (ih >= tensor.getShape()[2]) break; 

            for (int pw = 0; pw < poolSize; ++pw) {
                int iw = iwStart + pw;
                if (iw >= tensor.getShape()[3]) break;

                T val = tensor.at({n, ic, ih, iw});
                if (val > maxVal) {
                    maxVal = val;
                    maxIh = ih;
                    maxIw = iw;
                }
            }
        }

        return {maxVal, maxIh, maxIw};
    }
};

} // namespace smart_dnn

#endif // MAX_POOLING_2D_LAYER_HPP
