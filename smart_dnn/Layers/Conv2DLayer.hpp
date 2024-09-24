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

        // Store the input for use in backward pass
        this->input = input;

        int batchSize = input.shape()[0];
        int inputHeight = input.shape()[2];
        int inputWidth = input.shape()[3];

        int outputHeight = inputHeight - kernelHeight + 1;
        int outputWidth = inputWidth - kernelWidth + 1;

        Tensor output({batchSize, outputChannels, outputHeight, outputWidth}, 0.0f);

        // Iterate over batches and output channels
        for (int b = 0; b < batchSize; ++b) {
            for (int oc = 0; oc < outputChannels; ++oc) {
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        // Slice input region for convolution
                        Tensor inputSlice = input.slice({
                            {b, b + 1},               // Batch index
                            {0, inputChannels},        // All channels
                            {oh, oh + kernelHeight},   // Sliding window height
                            {ow, ow + kernelWidth}     // Sliding window width
                        });

                        // Slice the corresponding weight for the output channel
                        Tensor weightSlice = (*weights).slice({
                            {oc, oc + 1},              // Output channel index
                            {0, inputChannels},         // All input channels
                            {0, kernelHeight},          // Full kernel height
                            {0, kernelWidth}            // Full kernel width
                        });

                        float convResult = sum(inputSlice * weightSlice).at<float>(0);
                        convResult += (*biases).at<float>({static_cast<size_t>(oc), 0});
                        output.set({static_cast<size_t>(b), static_cast<size_t>(oc), static_cast<size_t>(oh), static_cast<size_t>(ow)}, convResult);

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
   Tensor backward(const Tensor& gradOutput) override {
        if (!input.has_value()) {
            throw std::runtime_error("Input not set. Forward pass must be called before backward.");
        }

        const Tensor& inputTensor = input.value();
        const Tensor& weightsTensor = *weights;
        Shape inputShape = inputTensor.shape();
        Shape weightsShape = weightsTensor.shape();
        Shape gradOutputShape = gradOutput.shape();

        // Initialize gradients
        this->weightGradients = zeros(weightsShape);
        this->biasGradients = zeros((*biases).shape());

        int batchSize = gradOutputShape[0];
        int outputHeight = gradOutputShape[2];
        int outputWidth = gradOutputShape[3];

        // Compute bias gradients by summing over the batch, output height, and width
        for (int b = 0; b < batchSize; ++b) {
            for (int oc = 0; oc < outputChannels; ++oc) {
                Tensor gradOutputSlice = gradOutput.slice({
                    {b, b + 1},  // Batch index
                    {oc, oc + 1},  // Output channel
                    {0, outputHeight},  // Full output height
                    {0, outputWidth}  // Full output width
                });
                (*biasGradients)[oc] += sum(gradOutputSlice);
            }
        }

        // Compute weight gradients by sliding over the input tensor
        for (int oc = 0; oc < outputChannels; ++oc) {
            for (int ic = 0; ic < inputChannels; ++ic) {
                for (int kh = 0; kh < kernelHeight; ++kh) {
                    for (int kw = 0; kw < kernelWidth; ++kw) {
                        Tensor res = Tensor({1}, 0.0f);
                        for (int b = 0; b < batchSize; ++b) {
                            for (int oh = 0; oh < outputHeight; ++oh) {
                                for (int ow = 0; ow < outputWidth; ++ow) {
                                    Tensor inputSlice = inputTensor.slice({
                                        {b, b + 1},  // Batch index
                                        {ic, ic + 1},  // Input channel
                                        {oh + kh, oh + kh + 1},  // Kernel height window
                                        {ow + kw, ow + kw + 1}  // Kernel width window
                                    });

                                    Tensor gradOutputSlice = gradOutput.slice({
                                        {b, b + 1},  // Batch index
                                        {oc, oc + 1},  // Output channel
                                        {oh, oh + 1},  // Output height
                                        {ow, ow + 1}  // Output width
                                    });
                                    res += sum(inputSlice * gradOutputSlice);
                                }
                            }
                        }
                        std::vector<size_t> indices = {static_cast<size_t>(oc), static_cast<size_t>(ic), static_cast<size_t>(kh), static_cast<size_t>(kw)};
                        weightGradients->set(indices, weightGradients->at<float>(indices) + res.at<float>(0));

                    }
                }
            }
        }

        // Compute input gradients
        Tensor gradInput(inputShape, 0.0f);
        for (int b = 0; b < batchSize; ++b) {
            for (int ic = 0; ic < inputChannels; ++ic) {
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        Tensor gradOutputSlice = gradOutput.slice({
                            {b, b + 1},  // Batch index
                            {0, outputChannels},  // All output channels
                            {oh, oh + 1},  // Output height
                            {ow, ow + 1}  // Output width
                        });

                        for (int kh = 0; kh < kernelHeight; ++kh) {
                            for (int kw = 0; kw < kernelWidth; ++kw) {
                                int ih = oh + kh;
                                int iw = ow + kw;

                                Tensor weightSlice = weightsTensor.slice({
                                    {0, outputChannels},  // All output channels
                                    {ic, ic + 1},  // Input channel
                                    {kh, kh + 1},  // Kernel height
                                    {kw, kw + 1}  // Kernel width
                                });
                                
                                std::vector<size_t> indices = {static_cast<size_t>(b), static_cast<size_t>(ic), static_cast<size_t>(ih), static_cast<size_t>(iw)};
                                gradInput.set(indices, gradInput.at<float>(indices) + sum(gradOutputSlice * weightSlice).at<float>(0));
                            }
                        }
                    }
                }
            }
        }

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
};

} // namespace sdnn

#endif // CONV2D_LAYER_HPP