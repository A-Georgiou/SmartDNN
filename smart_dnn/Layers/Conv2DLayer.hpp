#ifndef CONV2D_LAYER_HPP
#define CONV2D_LAYER_HPP

#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include "smart_dnn/Layer.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Optimizer.hpp"
#include "smart_dnn/Tensor/AdvancedTensorOperations.hpp"

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
        
        int outputHeight = (inputHeight + 2 * padding - dilation * (kernelHeight - 1) - 1) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - dilation * (kernelWidth - 1) - 1) / stride + 1;
        const int positions = outputHeight * outputWidth;

        TensorType colMatrix = im2col(input, kernelHeight, kernelWidth, stride, padding, dilation);
        TensorType weightMatrix = AdvancedTensorOperations<T>::reshape(*weights, {outputChannels, inputChannels * kernelHeight * kernelWidth});

        // Rows are output channels, columns are (batch, y, x) positions.
        TensorType output = AdvancedTensorOperations<T>::matmul(weightMatrix, colMatrix);

        const size_t rowSize = static_cast<size_t>(batchSize) * positions;
        T* out = output.getData().begin();
        for (int oc = 0; oc < outputChannels; ++oc) {
            const T bias = (*biases).at({oc, 0});
            for (size_t j = 0; j < rowSize; ++j) {
                out[oc * rowSize + j] += bias;
            }
        }

        return swapLeadingAxes(output, outputChannels, batchSize, positions, Shape({batchSize, outputChannels, outputHeight, outputWidth}));
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
        const Shape& inputShape = inputTensor.getShape();
        const int batchSize = inputShape[0];
        const int positions = gradOutput.getShape()[2] * gradOutput.getShape()[3];

        // Match the forward GEMM layout: rows are output channels, columns are (batch, y, x).
        TensorType gradOutput2D = swapLeadingAxes(gradOutput, batchSize, outputChannels, positions, Shape({outputChannels, batchSize * positions}));

        this->biasGradients = TensorType::zeros((*biases).getShape());
        const T* grad = gradOutput2D.getData().begin();
        const size_t rowSize = static_cast<size_t>(batchSize) * positions;
        for (int oc = 0; oc < outputChannels; ++oc) {
            biasGradients->at({oc, 0}) = std::accumulate(grad + oc * rowSize, grad + (oc + 1) * rowSize, T(0));
        }

        TensorType colMatrix = im2col(inputTensor, kernelHeight, kernelWidth, stride, padding, dilation);
        TensorType weightGradients2D = AdvancedTensorOperations<T>::matmul(gradOutput2D, AdvancedTensorOperations<T>::transpose(colMatrix, 1, 0));
        this->weightGradients = AdvancedTensorOperations<T>::reshape(weightGradients2D, (*weights).getShape());

        TensorType weightMatrix = AdvancedTensorOperations<T>::reshape(*weights, {outputChannels, inputChannels * kernelHeight * kernelWidth});
        TensorType gradCols = AdvancedTensorOperations<T>::matmul(AdvancedTensorOperations<T>::transpose(weightMatrix, 1, 0), gradOutput2D);
        return col2im(gradCols, inputShape, kernelHeight, kernelWidth, stride, padding, dilation);
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
        const size_t cols = colMatrix.getShape()[1];
        const T* in = input.getData().begin();
        T* col = colMatrix.getData().begin();

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
                                    col[w_row * cols + w_col] = in[flatIndex];
                                } else {
                                    col[w_row * cols + w_col] = 0;
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
        const size_t cols = colMatrix.getShape()[1];
        const T* col = colMatrix.getData().begin();
        T* out = output.getData().begin();

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
                                    out[flatIndex] += col[w_row * cols + w_col];
                                }
                            }
                        }
                    }
                }
            }
        }

        return output;
    }

    // Reorders contiguous [a, b, inner] data into [b, a, inner].
    static TensorType swapLeadingAxes(const TensorType& tensor, int a, int b, int inner, const Shape& resultShape) {
        TensorType result(resultShape);
        const T* src = tensor.getData().begin();
        T* dst = result.getData().begin();
        for (int i = 0; i < a; ++i) {
            for (int j = 0; j < b; ++j) {
                std::copy_n(src + (static_cast<size_t>(i) * b + j) * inner, inner, dst + (static_cast<size_t>(j) * a + i) * inner);
            }
        }
        return result;
    }
};

} // namespace smart_dnn

#endif // CONV2D_LAYER_HPP
