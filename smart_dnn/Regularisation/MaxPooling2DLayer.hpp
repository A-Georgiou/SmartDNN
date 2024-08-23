#ifndef MAX_POOLING_2D_LAYER_HPP
#define MAX_POOLING_2D_LAYER_HPP

#include "../Layer.hpp"
#include "../Tensor.hpp"
#include "../TensorOperations.hpp"
#include <algorithm>
#include <limits>
#include "../TensorWrapper.hpp"

class MaxPooling2DLayer : public Layer {
public:
    MaxPooling2DLayer(int poolSize): MaxPooling2DLayer(poolSize, poolSize) {}
    MaxPooling2DLayer(int poolSize, int stride) : poolSize(poolSize), stride(stride) {}

    Tensor forward(Tensor& input) override {
        if (input.shape().rank() != 4) {
            throw std::runtime_error("MaxPooling2DLayer: input tensor must have rank 4");
        }

        this->input = input;

        int batchSize = input.shape()[0];
        int inputChannels = input.shape()[1];
        int inputHeight = input.shape()[2];
        int inputWidth = input.shape()[3];

        int outputHeight = (inputHeight - poolSize) / stride + 1;
        int outputWidth = (inputWidth - poolSize) / stride + 1;

        Tensor output({batchSize, inputChannels, outputHeight, outputWidth});

        for (int n = 0; n < batchSize; ++n) {
            for (int ic = 0; ic < inputChannels; ++ic) {
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        float maxVal = -std::numeric_limits<float>::infinity();
                        for (int ph = 0; ph < poolSize; ++ph) {
                            for (int pw = 0; pw < poolSize; ++pw) {
                                int ih = oh * stride + ph;
                                int iw = ow * stride + pw;
                                if (ih < inputHeight && iw < inputWidth) {
                                    maxVal = std::max(maxVal, input({n, ic, ih, iw}));
                                }
                            }
                        }
                        output({n, ic, oh, ow}) = maxVal;
                    }
                }
            }
        }
        return output;
    }

    Tensor backward(Tensor& gradOutput) override {
        if(!input.valid()){
            throw std::runtime_error("MaxPooling2DLayer: input tensor is not set");
        }

        Tensor& tensorValue = *input;

        Tensor gradInput(tensorValue.shape(), 0.0f);

        int batchSize = tensorValue.shape()[0];
        int inputChannels = tensorValue.shape()[1];
        int inputHeight = tensorValue.shape()[2];
        int inputWidth = tensorValue.shape()[3];

        int outputHeight = gradOutput.shape()[2];
        int outputWidth = gradOutput.shape()[3];

        for (int n = 0; n < batchSize; ++n) {
            for (int ic = 0; ic < inputChannels; ++ic) {
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        float maxVal = -std::numeric_limits<float>::infinity();
                        int maxIh = 0;
                        int maxIw = 0;

                        for (int ph = 0; ph < poolSize; ++ph) {
                            for (int pw = 0; pw < poolSize; ++pw) {
                                int ih = oh * stride + ph;
                                int iw = ow * stride + pw;
                                if (ih < inputHeight && iw < inputWidth) {
                                    float val = tensorValue({n, ic, ih, iw});
                                    if (val > maxVal) {
                                        maxVal = val;
                                        maxIh = ih;
                                        maxIw = iw;
                                    }
                                }
                            }
                        }
                        gradInput({n, ic, maxIh, maxIw}) += gradOutput({n, ic, oh, ow});
                    }
                }
            }
        }
        return gradInput;
    }

    void updateWeights(Optimizer& optimizer) override {
            // No weights to update in max pooling layer.
    }

private:
    int poolSize;
    int stride;
    TensorWrapper input;
};

#endif // MAX_POOLING_2D_LAYER_HPP
