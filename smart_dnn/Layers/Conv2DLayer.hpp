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

template <typename T=float>
class Conv2DLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    Conv2DLayer(int inputChannels, int outputChannels, int squareKernalSize, int stride = 1, int padding = 0) 
        : Conv2DLayer(inputChannels, outputChannels, squareKernalSize, squareKernalSize, stride, padding) {}

    Conv2DLayer(int inputChannels, int outputChannels, int kernelHeight, int kernelWidth, int stride = 1, int padding = 0) 
        : kernelHeight(kernelHeight), kernelWidth(kernelWidth), stride(stride), padding(padding) {
        
        this->weights = TensorType::randn({outputChannels, inputChannels, kernelHeight, kernelWidth}, T(-1), T(1));
        this->biases = TensorType::zeros({outputChannels, 1});
    }

    TensorType forward(const TensorType& input) override {
        this->input = input;

        int batchSize = input.getShape()[0];
        int inputChannels = input.getShape()[1];
        int inputHeight = input.getShape()[2];
        int inputWidth = input.getShape()[3];
        int outputChannels = (*weights).getShape()[0];
        
        int outputHeight = (inputHeight - kernelHeight + 2 * padding) / stride + 1;
        int outputWidth = (inputWidth - kernelWidth + 2 * padding) / stride + 1;

        TensorType output({batchSize, outputChannels, outputHeight, outputWidth});

        TensorType colMatrix = im2col(input, kernelHeight, kernelWidth, stride, padding);
        
        TensorType weightMatrix = AdvancedTensorOperations<T>::reshape(*weights, {outputChannels, inputChannels * kernelHeight * kernelWidth});
        
        for (int n = 0; n < batchSize; ++n) {
            TensorType colMatrix_n = colMatrix.slice(0, n);
            TensorType output_n = AdvancedTensorOperations<T>::matmul(weightMatrix, colMatrix_n);
            output_n += (*biases);
            
            TensorType output_slice = AdvancedTensorOperations<T>::reshape(output_n, {outputChannels, outputHeight, outputWidth});
            output.slice(0, n) = output_slice;
        }
        return output;
    }


    TensorType backward(const TensorType& gradOutput) override {
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

        TensorType weightMatrix = AdvancedTensorOperations<T>::reshape(weightsTensor, {outputChannels, inputChannels * kernelHeight * kernelWidth});
        for (int n = 0; n < batchSize; ++n) {
            TensorType gradOutput_n = gradOutput.slice(0, n); // Slice once
            gradOutput_n = AdvancedTensorOperations<T>::reshape(gradOutput_n, {outputChannels, outputHeight * outputWidth});

            for (int oc = 0; oc < outputChannels; ++oc) {
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        biasGradTensor.at({oc, 0}) += gradOutput_n.at({oc, oh * outputWidth + ow});
                    }
                }
            }

            TensorType colMatrix_n = colMatrix.slice(0, n);
    
            weightGradTensor += AdvancedTensorOperations<T>::matmul(gradOutput_n, AdvancedTensorOperations<T>::transpose(colMatrix_n, 1, 0));

            TensorType dX_col = AdvancedTensorOperations<T>::transpose(AdvancedTensorOperations<T>::matmul(weightMatrix, gradOutput_n), 1, 0);
            gradInput.slice(0, n) = col2im(dX_col, inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, stride, padding);
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
        int batchSize = input.getShape()[0];
        int inputChannels = input.getShape()[1];
        int inputHeight = input.getShape()[2];
        int inputWidth = input.getShape()[3];

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

                                std::vector<int> indices = {n,
                                    ic * kernelHeight * kernelWidth + kh * kernelWidth + kw,
                                    colIndex};

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                    size_t flatIndex = n * inputStrides[0] +
                                                    ic * inputStrides[1] +
                                                    ih * inputStrides[2] +
                                                    iw * inputStrides[3];
                                    colMatrix.at(indices) = input.getData()[flatIndex];
                                } else {
                                    colMatrix.at(indices) = 0;
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

    TensorType col2im(const TensorType& colMatrix, int inputChannels, int inputHeight, int inputWidth, int kernelHeight, int kernelWidth, int stride, int padding) {
    int batchSize = colMatrix.getShape()[0];
    int outputHeight = (inputHeight - kernelHeight + 2 * padding) / stride + 1;
    int outputWidth = (inputWidth - kernelWidth + 2 * padding) / stride + 1;

    TensorType output({batchSize, inputChannels, inputHeight, inputWidth});
    output.getData().fill(T(0)); 

    const std::vector<int>& outputStrides = output.getShape().getStride();

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
                                std::vector<int> indices = {n, ic * kernelHeight * kernelWidth + kh * kernelWidth + kw,
                                                                colIndex};
                                size_t flatIndex = n * outputStrides[0] +
                                                   ic * outputStrides[1] +
                                                   ih * outputStrides[2] +
                                                   iw * outputStrides[3];

                                output.getData()[flatIndex] += colMatrix.at(indices);
                            }
                        }
                    }
                }
                ++colIndex;
            }
        }
    }

    return output;
}
};

} // namespace smart_dnn

#endif // CONV2D_LAYER_HPP
