#ifndef CONV2D_LAYER_HPP
#define CONV2D_LAYER_HPP

#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/Optimizer.hpp"
#include "smart_dnn/Layer.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>

namespace sdnn {

class Conv2DLayer : public Layer {
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
    Tensor forward(const Tensor& input) override {
        assert(input.shape().rank() == 4 && "Input must be 4D (batch, channels, height, width)");
        assert(input.shape()[1] == inputChannels && "Input channels must match layer's input channels");

        this->input = input;

        int batchSize = input.shape()[0];
        int inputHeight = input.shape()[2];
        int inputWidth = input.shape()[3];

        int outputHeight = (inputHeight - kernelHeight + 2 * padding) / stride + 1;
        int outputWidth = (inputWidth - kernelWidth + 2 * padding) / stride + 1;

        Tensor cols = im2col(input, kernelHeight, kernelWidth, stride, padding);
        Tensor weightsMat = reshape((*weights), {outputChannels, inputChannels * kernelHeight * kernelWidth});
        Tensor outputMat = matmul(weightsMat, cols);

        Tensor biasesMat = outputMat + reshape((*biases), {outputChannels, 1});
        
        return reshape(biasesMat, {batchSize, outputChannels, outputHeight, outputWidth});
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (!input.has_value()) {
            throw std::runtime_error("Input not set. Forward pass must be called before backward.");
        }

        const Tensor& inputTensor = input.value();
        int batchSize = inputTensor.shape()[0];
        int inputHeight = inputTensor.shape()[2];
        int inputWidth = inputTensor.shape()[3];

        int outputHeight = gradOutput.shape()[2];
        int outputWidth = gradOutput.shape()[3];

        Tensor gradOutputMat = reshape(gradOutput, {outputChannels, batchSize * outputHeight * outputWidth});

        // Compute weight gradients
        Tensor cols = im2col(inputTensor, kernelHeight, kernelWidth, stride, padding);
        // Transpose cols with axes {1, 0}
        Tensor colsTransposed = transpose(cols, {1, 0});
        weightGradients = reshape(matmul(gradOutputMat, colsTransposed), weights->shape());

        // Compute bias gradients
        biasGradients = reshape(sum(gradOutputMat, {1}), {outputChannels, 1});

        Tensor weightsMat = reshape((*weights), {outputChannels, inputChannels * kernelHeight * kernelWidth});
        Tensor weightsMatTransposed = transpose(weightsMat, {1, 0});
        Tensor gradCols = matmul(weightsMatTransposed, gradOutputMat);
        Tensor gradInput = col2im(gradCols, inputTensor.shape(), kernelHeight, kernelWidth, stride, padding);

        return gradInput;
    }

    void updateWeights(Optimizer& optimizer) override {
        if (!weights || !biases || !weightGradients || !biasGradients) {
            throw std::runtime_error("Weights or gradients are not initialized!");
        }

        optimizer.optimize({std::ref(*weights), std::ref(*biases)},
                           {std::ref(*weightGradients), std::ref(*biasGradients)});
    }

    /*
    
        Test helper functions
    
    */

    Tensor getWeights() const {
        return *weights;
    }

    Tensor getBiases() const {
        return *biases;
    }

    Tensor getWeightGradients() const {
        return *weightGradients;
    }

    Tensor getBiasGradients() const {
        return *biasGradients;
    }

    void setWeights(const Tensor& newWeights) {
        weights = newWeights;
    }

    void setBiases(const Tensor& newBiases) {
        biases = newBiases;
    }

private:
    std::optional<Tensor> weights;
    std::optional<Tensor> biases;
    std::optional<Tensor> input;
    std::optional<Tensor> weightGradients;
    std::optional<Tensor> biasGradients;
    
    int inputChannels;
    int outputChannels;
    int kernelHeight;
    int kernelWidth;
    int stride;
    int padding;
    int dilation;

    void initializeWeights() {
        weights = rand({outputChannels, inputChannels, kernelHeight, kernelWidth}, dtype::f32);
        biases = zeros({outputChannels, 1}, dtype::f32);
    }

    Tensor im2col(const Tensor& input, int kernelHeight, int kernelWidth, int stride, int padding) {
        int batchSize = input.shape()[0];
        int channels = input.shape()[1];
        int height = input.shape()[2];
        int width = input.shape()[3];

        int outputHeight = (height - kernelHeight + 2 * padding) / stride + 1;
        int outputWidth = (width - kernelWidth + 2 * padding) / stride + 1;

        Tensor cols({channels * kernelHeight * kernelWidth, batchSize * outputHeight * outputWidth}, 0.0f);

        for (int c = 0; c < channels; ++c) {
            for (int kh = 0; kh < kernelHeight; ++kh) {
                for (int kw = 0; kw < kernelWidth; ++kw) {
                    int rowIdx = c * kernelHeight * kernelWidth + kh * kernelWidth + kw;
                    for (int h = 0; h < outputHeight; ++h) {
                        for (int w = 0; w < outputWidth; ++w) {
                            int hPad = h * stride - padding + kh;
                            int wPad = w * stride - padding + kw;
                            if (hPad >= 0 && hPad < height && wPad >= 0 && wPad < width) {
                                for (int n = 0; n < batchSize; ++n) {
                                    int colIdx = (n * outputHeight + h) * outputWidth + w;
                                    cols.set({static_cast<size_t>(rowIdx), static_cast<size_t>(colIdx)},
                                             input.at<float>({static_cast<size_t>(n), static_cast<size_t>(c),
                                                              static_cast<size_t>(hPad), static_cast<size_t>(wPad)}));
                                }
                            }
                        }
                    }
                }
            }
        }
        return cols;
    }

    Tensor col2im(const Tensor& cols, const Shape& inputShape, int kernelHeight, int kernelWidth, int stride, int padding) {
        int batchSize = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int outputHeight = (height - kernelHeight + 2 * padding) / stride + 1;
        int outputWidth = (width - kernelWidth + 2 * padding) / stride + 1;

        Tensor output(inputShape, 0.0f);

        for (int c = 0; c < channels; ++c) {
            for (int kh = 0; kh < kernelHeight; ++kh) {
                for (int kw = 0; kw < kernelWidth; ++kw) {
                    int rowIdx = c * kernelHeight * kernelWidth + kh * kernelWidth + kw;
                    for (int h = 0; h < outputHeight; ++h) {
                        for (int w = 0; w < outputWidth; ++w) {
                            int hPad = h * stride - padding + kh;
                            int wPad = w * stride - padding + kw;
                            if (hPad >= 0 && hPad < height && wPad >= 0 && wPad < width) {
                                for (int n = 0; n < batchSize; ++n) {
                                    int colIdx = (n * outputHeight + h) * outputWidth + w;
                                    float value = cols.at<float>({static_cast<size_t>(rowIdx), static_cast<size_t>(colIdx)});
                                    output.set({static_cast<size_t>(n), static_cast<size_t>(c),
                                                static_cast<size_t>(hPad), static_cast<size_t>(wPad)},
                                               output.at<float>({static_cast<size_t>(n), static_cast<size_t>(c),
                                                                 static_cast<size_t>(hPad), static_cast<size_t>(wPad)}) + value);
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

}; // namespace sdnn

#endif // CONV2D_LAYER_HPP