#ifndef CONV2D_LAYER_HPP
#define CONV2D_LAYER_HPP

#include <vector>
#include <cmath>
#include <iostream>
#include "../Layer.hpp"
#include "../Tensor.hpp"
#include "../Optimizer.hpp"
#include "../TensorOperations.hpp"
#include "../TensorWrapper.hpp"

class Conv2DLayer : public Layer {
public:
    Conv2DLayer(int inputChannels, int outputChannels, int squareKernalSize, int stride = 1, int padding = 0) 
        : Conv2DLayer(inputChannels, outputChannels, squareKernalSize, squareKernalSize, stride, padding) {}

    Conv2DLayer(int inputChannels, int outputChannels, int kernelHeight, int kernelWidth, int stride = 1, int padding = 0) 
        : kernelHeight(kernelHeight), kernelWidth(kernelWidth), stride(stride), padding(padding) {
        
        this->weights = Tensor({outputChannels, inputChannels, kernelHeight, kernelWidth});
        this->biases = Tensor({outputChannels, 1});

        (weights.get()).randomize(-1.0f, 1.0f);
        (biases.get()).fill(0.0f);
    }

    Tensor forward(Tensor& input) override {
        this->input = input;

        Tensor& weights = this->weights.get();
        Tensor& biases = this->biases.get();

        int batchSize = input.shape()[0];
        int inputChannels = input.shape()[1];
        int inputHeight = input.shape()[2];
        int inputWidth = input.shape()[3];
        int outputChannels = weights.shape()[0];
        
        int outputHeight = (inputHeight - kernelHeight + 2 * padding) / stride + 1;
        int outputWidth = (inputWidth - kernelWidth + 2 * padding) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0) {
            throw std::runtime_error("Invalid output shape calculated in Conv2DLayer. Ensure kernel size, padding, and stride are set correctly.");
        }

        Tensor output({batchSize, outputChannels, outputHeight, outputWidth});

        for (int n = 0; n < batchSize; ++n) {
            for (int oc = 0; oc < outputChannels; ++oc) {
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        float value = biases({oc, 0});

                        for (int ic = 0; ic < inputChannels; ++ic) {
                            for (int kh = 0; kh < kernelHeight; ++kh) {
                                for (int kw = 0; kw < kernelWidth; ++kw) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
                                    
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                        value += input({n, ic, ih, iw}) * weights({oc, ic, kh, kw});
                                    }
                                }
                            }
                        }
                        output({n, oc, oh, ow}) = value;
                    }
                }
            }
        }

        return output;
    }

    Tensor backward(Tensor& gradOutput) override {
        Tensor& inputTensor = (*input);
        Tensor& weightsTensor = (*weights);
        Tensor& biasesTensor = (*biases);

        Shape inputShape = inputTensor.shape();
        Shape weightsShape = weightsTensor.shape();
        
        int batchSize = inputShape[0];
        int inputChannels = inputShape[1];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];
        int outputChannels = weightsTensor.shape()[0];
        
        int outputHeight = (inputHeight - kernelHeight + 2 * padding) / stride + 1;
        int outputWidth = (inputWidth - kernelWidth + 2 * padding) / stride + 1;

        weightGradients = Tensor(weightsShape);
        biasGradients = Tensor(biasesTensor.shape());
        Tensor weightGradTensor = (*weightGradients);
        Tensor biasGradTensor = (*biasGradients);
        Tensor gradInput(inputShape);

        for (int n = 0; n < batchSize; ++n) {
            for (int oc = 0; oc < outputChannels; ++oc) {
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        float grad = gradOutput({n, oc, oh, ow});
                        biasGradTensor({oc, 0}) += grad;

                        for (int ic = 0; ic < inputChannels; ++ic) {
                            for (int kh = 0; kh < kernelHeight; ++kh) {
                                for (int kw = 0; kw < kernelWidth; ++kw) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;

                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                        weightGradTensor({oc, ic, kh, kw}) += inputTensor({n, ic, ih, iw}) * grad;
                                        gradInput({n, ic, ih, iw}) += weightsTensor({oc, ic, kh, kw}) * grad;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return gradInput;
    }

    void updateWeights(Optimizer& optimizer) override {
        optimizer.optimize({std::ref(*weights),
                            std::ref(*biases)},
                            {std::ref(*weightGradients),
                            std::ref(*biasGradients)});
    }

private:
    TensorWrapper weights;
    TensorWrapper biases;
    TensorWrapper input;
    TensorWrapper weightGradients;
    TensorWrapper biasGradients;
    
    int kernelHeight;
    int kernelWidth;
    int stride;
    int padding;
};

#endif // CONV2D_LAYER_HPP
