#ifndef CONV2D_LAYER_HPP
#define CONV2D_LAYER_HPP

#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>
#include "smart_dnn/Layer.hpp"
#include "smart_dnn/tensor/Tensor.hpp"
#include "smart_dnn/Optimizer.hpp"
#include "smart_dnn/tensor/AdvancedTensorOperations.hpp"

namespace smart_dnn {

template <typename T=float>
class Conv2DLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    Conv2DLayer(int inputChannels, int outputChannels, int squareKernalSize, int stride = 1, int padding = 0, int dilation = 1) 
        : Conv2DLayer(inputChannels, outputChannels, squareKernalSize, squareKernalSize, stride, padding, dilation) {}

    Conv2DLayer(int inputChannels, int outputChannels, int kernelHeight, int kernelWidth, int stride = 1, int padding = 0, int dilation = 1) 
        : inputChannels(inputChannels), outputChannels(outputChannels), kernelHeight(kernelHeight), kernelWidth(kernelWidth), 
          stride(stride), padding(padding), dilation(dilation) {
        
        initializeWeights();
    }

    /*
    
    Input: 4D tensor (batchSize, inputChannels, inputHeight, inputWidth)
    Output: 4D tensor (batchSize, outputChannels, outputHeight, outputWidth)
    
    */
    TensorType forward(const TensorType& input) override {
        assert(input.getShape().rank() == 4 && "Input must be 4D (batch, channels, height, width)");
        assert(input.getShape()[1] == inputChannels && "Input channels must match layer's input channels");

        // Store the input for use in backward pass
        this->input = input;

        int batchSize = input.getShape()[0];
        int inputHeight = input.getShape()[2];
        int inputWidth = input.getShape()[3];
        
        int outputHeight = inputHeight - kernelHeight + 1;
        int outputWidth = inputWidth - kernelWidth + 1;

        TensorType output({batchSize, outputChannels, outputHeight, outputWidth});

        const auto& weightData = weights->getData();
        const auto& biasData = biases->getData();
        const auto& inputData = input.getData();

        for (int b = 0; b < batchSize; ++b) {
            for (int oc = 0; oc < outputChannels; ++oc) {
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        float sum = 0.0f;
                        for (int kh = 0; kh < kernelHeight; ++kh) {
                            for (int kw = 0; kw < kernelWidth; ++kw) {
                                int ih = oh + kh;
                                int iw = ow + kw;
                                sum += inputData.at({b, 0, ih, iw}) * 
                                    weightData.at({oc, 0, kh, kw});
                            }
                        }
                        sum += biasData.at({oc, 0});
                        output.at({b, oc, oh, ow}) = sum;
                    }
                }
            }
        }

        return output;
    }

    /*
    
    Input: 4D tensor (batchSize, outputChannels, outputHeight, outputWidth)
    Output: 4D tensor (batchSize, inputChannels, inputHeight, inputWidth)
    
    */
   TensorType backward(const TensorType& gradOutput) override {
        if (!input.has_value()) {
            throw std::runtime_error("Input not set. Forward pass must be called before backward.");
        }

        const TensorType& inputTensor = input.value();
        const TensorType& weightsTensor = *weights;

        Shape inputShape = inputTensor.getShape();
        Shape weightsShape = weightsTensor.getShape();
        Shape gradOutputShape = gradOutput.getShape();

        // Initialize gradients
        this->weightGradients = TensorType::zeros(weightsShape);
        this->biasGradients = TensorType::zeros((*biases).getShape());

        // Compute bias gradients
        for (int oh = 0; oh < gradOutputShape[2]; ++oh) {
            for (int ow = 0; ow < gradOutputShape[3]; ++ow) {
                biasGradients->getData()[0] += gradOutput.at({0, 0, oh, ow});
            }
        }

        // Compute weight gradients
        for (int kh = 0; kh < kernelHeight; ++kh) {
            for (int kw = 0; kw < kernelWidth; ++kw) {
                float sum = 0.0f;
                for (int oh = 0; oh < gradOutputShape[2]; ++oh) {
                    for (int ow = 0; ow < gradOutputShape[3]; ++ow) {
                        sum += inputTensor.at({0, 0, oh+kh, ow+kw}) * gradOutput.at({0, 0, oh, ow});
                    }
                }
                weightGradients->at({0, 0, kh, kw}) = sum;
            }
        }

        // Compute input gradients
        TensorType gradInput(inputShape, 0.0f);
        for (int oh = 0; oh < gradOutputShape[2]; ++oh) {
            for (int ow = 0; ow < gradOutputShape[3]; ++ow) {
                for (int kh = 0; kh < kernelHeight; ++kh) {
                    for (int kw = 0; kw < kernelWidth; ++kw) {
                        int ih = oh + kh;
                        int iw = ow + kw;
                        gradInput.at({0, 0, ih, iw}) += weightsTensor.at({0, 0, kh, kw}) * gradOutput.at({0, 0, oh, ow});
                    }
                }
            }
        }

        return gradInput;
    }

    void updateWeights(Optimizer<T>& optimizer) override {
        if (!weights || !biases || !weightGradients || !biasGradients) {
            throw std::runtime_error("Weights or gradients are not initialized!");
        }

        optimizer.optimize({std::ref(*weights), std::ref(*biases)},
                        {std::ref(*weightGradients), std::ref(*biasGradients)});
    }

    /*
    
        Test helper functions
    
    */

    TensorType getWeights() const {
        return *weights;
    }

    TensorType getBiases() const {
        return *biases;
    }

    TensorType getWeightGradients() const {
        return *weightGradients;
    }

    TensorType getBiasGradients() const {
        return *biasGradients;
    }

    void setWeights(const TensorType& newWeights) {
        weights = newWeights;
    }

    void setBiases(const TensorType& newBiases) {
        biases = newBiases;
    }

private:
    std::optional<TensorType> weights;
    std::optional<TensorType> biases;
    std::optional<TensorType> input;
    std::optional<TensorType> weightGradients;
    std::optional<TensorType> biasGradients;
    
    int inputChannels;
    int outputChannels;
    int kernelHeight;
    int kernelWidth;
    int stride;
    int padding;
    int dilation;

    void initializeWeights() {
        weights = TensorType::rand({outputChannels, inputChannels, kernelHeight, kernelWidth});
        biases = TensorType::zeros({outputChannels, 1});
    }

    TensorType im2col(const TensorType& input, int kernelHeight, int kernelWidth, int stride, int padding, int dilation) {
        int batchSize = input.getShape()[0];
        int inputChannels = input.getShape()[1];
        int inputHeight = input.getShape()[2];
        int inputWidth = input.getShape()[3];

        int outputHeight = (inputHeight + 2 * padding - dilation * (kernelHeight - 1) - 1) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - dilation * (kernelWidth - 1) - 1) / stride + 1;

        TensorType colMatrix({inputChannels * kernelHeight * kernelWidth, outputHeight * outputWidth * batchSize});

        const std::vector<size_t>& inputStrides = input.getShape().getStride();

        #pragma omp parallel for collapse(2)
        for (int n = 0; n < batchSize; ++n) {
            for (int c = 0; c < inputChannels; ++c) {
                for (int kh = 0; kh < kernelHeight; ++kh) {
                    for (int kw = 0; kw < kernelWidth; ++kw) {
                        int w_row = c * kernelHeight * kernelWidth + kh * kernelWidth + kw;
                        for (int oh = 0; oh < outputHeight; ++oh) {
                            for (int ow = 0; ow < outputWidth; ++ow) {
                                int ih = oh * stride + kh * dilation - padding;
                                int iw = ow * stride + kw * dilation - padding;
                                int w_col = (n * outputHeight + oh) * outputWidth + ow;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                    size_t flatIndex = n * inputStrides[0] + c * inputStrides[1] + 
                                                       ih * inputStrides[2] + iw * inputStrides[3];
                                    colMatrix.at({w_row, w_col}) = input.getData()[flatIndex];
                                } else {
                                    colMatrix.at({w_row, w_col}) = 0;
                                }
                            }
                        }
                    }
                }
            }
        }

        return colMatrix;
    }

    TensorType col2im(const TensorType& colMatrix, const Shape& outputShape, int kernelHeight, int kernelWidth, int stride, int padding, int dilation) {
        int batchSize = outputShape[0];
        int inputChannels = outputShape[1];
        int inputHeight = outputShape[2];
        int inputWidth = outputShape[3];

        int outputHeight = (inputHeight + 2 * padding - dilation * (kernelHeight - 1) - 1) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - dilation * (kernelWidth - 1) - 1) / stride + 1;

        TensorType output(outputShape);
        output.getData().fill(T(0));

        const std::vector<size_t>& outputStrides = output.getShape().getStride();

        #pragma omp parallel for collapse(2)
        for (int n = 0; n < batchSize; ++n) {
            for (int c = 0; c < inputChannels; ++c) {
                for (int kh = 0; kh < kernelHeight; ++kh) {
                    for (int kw = 0; kw < kernelWidth; ++kw) {
                        int w_row = c * kernelHeight * kernelWidth + kh * kernelWidth + kw;
                        for (int oh = 0; oh < outputHeight; ++oh) {
                            for (int ow = 0; ow < outputWidth; ++ow) {
                                int ih = oh * stride + kh * dilation - padding;
                                int iw = ow * stride + kw * dilation - padding;
                                int w_col = (n * outputHeight + oh) * outputWidth + ow;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                    size_t flatIndex = n * outputStrides[0] + c * outputStrides[1] +
                                                       ih * outputStrides[2] + iw * outputStrides[3];
                                    output.getData()[flatIndex] += colMatrix.at({w_row, w_col});
                                }
                            }
                        }
                    }
                }
            }
        }

        return output;
    }
};

} // namespace smart_dnn

#endif // CONV2D_LAYER_HPP
