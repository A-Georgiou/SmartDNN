#ifndef MAX_POOLING_2D_LAYER_HPP
#define MAX_POOLING_2D_LAYER_HPP

#include <algorithm>
#include <limits>
#include "smart_dnn/Layer.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"

namespace sdnn {

class MaxPooling2DLayer : public Layer {
public:
    MaxPooling2DLayer(int poolSize) : MaxPooling2DLayer(poolSize, poolSize, 0) {}
    MaxPooling2DLayer(int poolSize, int stride) : MaxPooling2DLayer(poolSize, stride, 0) {}
    MaxPooling2DLayer(int poolSize, int stride, int padding) : poolSize(poolSize), stride(stride), padding(padding) {}

    Tensor forward(const Tensor& input) override {
        if (input.shape().rank() != 4) {
            throw std::runtime_error("MaxPooling2DLayer: input tensor must have rank 4");
        }

        this->input = input;

        int batchSize = input.shape()[0];
        int inputChannels = input.shape()[1];  // Number of channels should remain unchanged
        int inputHeight = input.shape()[2];
        int inputWidth = input.shape()[3];

        // Calculate output dimensions based on stride and pooling size
        int outputHeight = (inputHeight + 2 * padding - poolSize) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - poolSize) / stride + 1;

        // Output shape retains the number of channels
        Tensor output = zeros({batchSize, inputChannels, outputHeight, outputWidth}, input.type());

        for (int n = 0; n < batchSize; ++n) {
            for (int ic = 0; ic < inputChannels; ++ic) {  // Ensure channel is retained
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        std::cout << "finding max in pool window" << std::endl;
                        auto [maxVal, maxIh, maxIw] = findMaxInPoolWindow(input, n, ic, oh, ow);
                        std::cout << "maxVal: " << maxVal << ", maxIh: " << maxIh << ", maxIw: " << maxIw << std::endl;
                        output.set({static_cast<size_t>(n), static_cast<size_t>(ic), static_cast<size_t>(oh), static_cast<size_t>(ow)}, maxVal);
                    }
                }
            }
        }
        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (!input.has_value()) {
            throw std::runtime_error("MaxPooling2DLayer: input tensor is not set");
        }

        Tensor& tensorValue = *input;
        Tensor gradInput = zeros(tensorValue.shape(), tensorValue.type());

        int batchSize = tensorValue.shape()[0];
        int inputChannels = tensorValue.shape()[1];
        int outputHeight = gradOutput.shape()[2];
        int outputWidth = gradOutput.shape()[3];

        for (int n = 0; n < batchSize; ++n) {
            for (int ic = 0; ic < inputChannels; ++ic) {
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        auto [maxVal, maxIh, maxIw] = findMaxInPoolWindow(tensorValue, n, ic, oh, ow);
                        gradInput.set({static_cast<size_t>(n), static_cast<size_t>(ic), static_cast<size_t>(maxIh), static_cast<size_t>(maxIw)},
                                gradOutput.at<float>({static_cast<size_t>(n), static_cast<size_t>(ic), static_cast<size_t>(oh), static_cast<size_t>(ow)}));
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
    std::optional<Tensor> input;

    std::tuple<float, int, int> findMaxInPoolWindow(const Tensor& tensor, int n, int ic, int oh, int ow) {
        float maxVal = -std::numeric_limits<float>::infinity();
        int maxIh = -1, maxIw = -1;
        int inputHeight = tensor.shape()[2];
        int inputWidth = tensor.shape()[3];

        int ihStart = oh * stride - padding;
        int iwStart = ow * stride - padding;

        for (int ph = 0; ph < poolSize; ++ph) {
            for (int pw = 0; pw < poolSize; ++pw) {
                int ih = ihStart + ph;
                int iw = iwStart + pw;

                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                    float val = tensor.at<float>({static_cast<size_t>(n), static_cast<size_t>(ic), static_cast<size_t>(ih), static_cast<size_t>(iw)});
                    if (val > maxVal) {
                        maxVal = val;
                        maxIh = ih;
                        maxIw = iw;
                    }
                }
            }
        }

        return {maxVal, maxIh, maxIw};
    }
};

} // namespace sdnn

#endif // MAX_POOLING_2D_LAYER_HPP
