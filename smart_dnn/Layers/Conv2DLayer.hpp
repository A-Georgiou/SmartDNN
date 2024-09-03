#ifndef CONV2D_LAYER_HPP
#define CONV2D_LAYER_HPP

#include <vector>
#include <cmath>
#include <iostream>
#include "../Layer.hpp"
#include "../Tensor/Tensor.hpp"
#include "../Optimizer.hpp"
#include "../Tensor/AdvancedTensorOperations.hpp"

namespace smart_dnn {

template <typename T>
class Conv2DLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    Conv2DLayer(int inputChannels, int outputChannels, int squareKernalSize, int stride = 1, int padding = 0) 
        : Conv2DLayer(inputChannels, outputChannels, squareKernalSize, squareKernalSize, stride, padding) {}

    Conv2DLayer(int inputChannels, int outputChannels, int kernelHeight, int kernelWidth, int stride = 1, int padding = 0) 
        : kernelHeight(kernelHeight), kernelWidth(kernelWidth), stride(stride), padding(padding) {
        
        this->weights = Tensor::randn({outputChannels, inputChannels, kernelHeight, kernelWidth}, -1.0f, 1.0f);
        this->biases = Tensor::zeros({outputChannels, 1});
    }

    TensorType forward(TensorType& input) override {
        this->input = input;

        int batchSize = input.shape()[0];
        int inputChannels = input.shape()[1];
        int inputHeight = input.shape()[2];
        int inputWidth = input.shape()[3];
        int outputChannels = weights.shape()[0];
        
        int outputHeight = (inputHeight - kernelHeight + 2 * padding) / stride + 1;
        int outputWidth = (inputWidth - kernelWidth + 2 * padding) / stride + 1;

        TensorType output({batchSize, outputChannels, outputHeight, outputWidth});

        TensorType colMatrix = im2col(input, kernelHeight, kernelWidth, stride, padding);
        
        TensorType weightMatrix = weights.reshape({outputChannels, inputChannels * kernelHeight * kernelWidth});
        
        for (int n = 0; n < batchSize; ++n) {
            TensorType output_n = weightMatrix.matmul(colMatrix[n]);
            output_n = output_n.add(biases);
            output[n] = output_n.reshape({outputChannels, outputHeight, outputWidth});
        }

        return output;
    }


    TensorType backward(TensorType& gradOutput) override {
        TensorType& inputTensor = (*input);
        TensorType& weightsTensor = (*weights);
        TensorType& biasesTensor = (*biases);

        Shape inputShape = inputTensor.getShape();
        Shape weightsShape = weightsTensor.getShape();

        int batchSize = inputShape[0];
        int inputChannels = inputShape[1];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];
        int outputChannels = weightsTensor.getShape()[0];

        int outputHeight = (inputHeight - kernelHeight + 2 * padding) / stride + 1;
        int outputWidth = (inputWidth - kernelWidth + 2 * padding) / stride + 1;

        // Initialize gradients
        weightGradients = TensorType(weightsShape);
        biasGradients = TensorType(biasesTensor.getShape());
        TensorType weightGradTensor = (*weightGradients);
        TensorType biasGradTensor = (*biasGradients);
        TensorType gradInput(inputShape);

        // Compute bias gradients
        for (int n = 0; n < batchSize; ++n) {
            for (int oc = 0; oc < outputChannels; ++oc) {
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        biasGradTensor.at({oc, 0}) += gradOutput.at({n, oc, oh, ow});
                    }
                }
            }
        }

        // Compute dW (weight gradients) using im2col
        TensorType colMatrix = im2col(inputTensor, kernelHeight, kernelWidth, stride, padding);

        for (int n = 0; n < batchSize; ++n) {
            TensorType gradOutput_n = gradOutput.slice(n); // Get the nth output gradient
            gradOutput_n = gradOutput_n.reshape({outputChannels, outputHeight * outputWidth});

            // Matrix multiplication: dW = gradOutput_n * colMatrix[n].T
            weightGradTensor += gradOutput_n.matmul(colMatrix[n].transpose());
        }

        // Compute dX (input gradients) using col2im
        TensorType weightMatrix = weightsTensor.reshape({outputChannels, inputChannels * kernelHeight * kernelWidth});
        for (int n = 0; n < batchSize; ++n) {
            TensorType gradOutput_n = gradOutput.slice(n); // Get the nth output gradient
            gradOutput_n = gradOutput_n.reshape({outputChannels, outputHeight * outputWidth});

            // Matrix multiplication: dX_col = weightMatrix.T * gradOutput_n
            TensorType dX_col = weightMatrix.transpose().matmul(gradOutput_n);

            // Use col2im to fold dX_col back into the shape of the input
            gradInput.slice(n) = col2im(dX_col, inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, stride, padding);
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

private:
    std::optional<TensorType> weights;
    std::optional<TensorType> biases;
    std::optional<TensorType> input;
    std::optional<TensorType> weightGradients;
    std::optional<TensorType> biasGradients;
    
    int kernelHeight;
    int kernelWidth;
    int stride;
    int padding;

   TensorType im2col(const TensorType& input, int kernelHeight, int kernelWidth, int stride, int padding) {
        int batchSize = input.shape()[0];
        int inputChannels = input.shape()[1];
        int inputHeight = input.shape()[2];
        int inputWidth = input.shape()[3];

        int outputHeight = (inputHeight - kernelHeight + 2 * padding) / stride + 1;
        int outputWidth = (inputWidth - kernelWidth + 2 * padding) / stride + 1;

        TensorType colMatrix({batchSize, inputChannels * kernelHeight * kernelWidth, outputHeight * outputWidth});

        const std::vector<int>& inputStrides = input.getShape().getStride();

        for (int n = 0; n < batchSize; ++n) {
            int colIndex = 0;
            for (int oh = 0; oh < outputHeight; ++oh) {
                for (int ow = 0; ow < outputWidth; ++ow) {
                    for (int ic = 0; ic < inputChannels; ++ic) {
                        for (int kh = 0; kh < kernelHeight; ++kh) {
                            for (int kw = 0; kw < kernelWidth; ++kw) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                    size_t flatIndex = n * inputStrides[0] +
                                                    ic * inputStrides[1] +
                                                    ih * inputStrides[2] +
                                                    iw * inputStrides[3];

                                    colMatrix.at({n, ic * kernelHeight * kernelWidth + kh * kernelWidth + kw, colIndex}) = 
                                        input.data()[flatIndex];
                                } else {
                                    colMatrix.at({n, ic * kernelHeight * kernelWidth + kh * kernelWidth + kw, colIndex}) = 0;
                                }
                            }
                        }
                    }
                    ++colIndex;
                }
            }
        }

        return colMatrix;
    }
};

} // namespace smart_dnn

#endif // CONV2D_LAYER_HPP
