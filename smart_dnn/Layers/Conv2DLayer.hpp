#ifndef CONV2D_LAYER_HPP
#define CONV2D_LAYER_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <cassert>
#include "../Layer.hpp"
#include "../Tensor/Tensor.hpp"
#include "../Optimizer.hpp"
#include "../Tensor/AdvancedTensorOperations.hpp"
#include "../RandomEngine.hpp"

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

        this->input = input;

        int batchSize = input.getShape()[0];
        int inputHeight = input.getShape()[2];
        int inputWidth = input.getShape()[3];
        
        int outputHeight = (inputHeight + 2 * padding - dilation * (kernelHeight - 1) - 1) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - dilation * (kernelWidth - 1) - 1) / stride + 1;
        // Compute im2col
        TensorType colMatrix = im2col(input, kernelHeight, kernelWidth, stride, padding, dilation);

        TensorType weightMatrix = AdvancedTensorOperations<T>::reshape(*weights, {outputChannels, inputChannels * kernelHeight * kernelWidth});

        // Reshape colMatrix
        colMatrix = AdvancedTensorOperations<T>::reshape(colMatrix, {inputChannels * kernelHeight * kernelWidth, outputHeight * outputWidth * batchSize});

        // Rows are output channels, columns are (batch, y, x) positions.
        TensorType output = AdvancedTensorOperations<T>::matmul(weightMatrix, colMatrix);

        const int positions = outputHeight * outputWidth;
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
        TensorType& inputTensor = (*input);
        TensorType& weightsTensor = (*weights);

        Shape inputShape = inputTensor.getShape();
        Shape weightsShape = weightsTensor.getShape();

        int batchSize = inputShape[0];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];

        int outputHeight = gradOutput.getShape()[2];  // Correct shape (height)
        int outputWidth = gradOutput.getShape()[3];   // Correct shape (width)

        // Initialize gradients
        weightGradients = TensorType::rand(weightsShape);
        biasGradients = TensorType::zeros((*biases).getShape());

        TensorType gradInput(inputShape);

        // Match the forward GEMM layout: rows are output channels, columns are (batch, y, x).
        const int positions = outputHeight * outputWidth;
        TensorType gradOutput_reshaped = swapLeadingAxes(gradOutput, batchSize, outputChannels, positions, Shape({outputChannels, batchSize * positions}));

        const T* grad = gradOutput_reshaped.getData().begin();
        const size_t rowSize = static_cast<size_t>(batchSize) * positions;
        for (int oc = 0; oc < outputChannels; ++oc) {
            biasGradients->at({oc, 0}) = std::accumulate(grad + oc * rowSize, grad + (oc + 1) * rowSize, T(0));
        }

        // Compute dW (weight gradients) using im2col
        TensorType colMatrix = im2col(inputTensor, kernelHeight, kernelWidth, stride, padding, dilation);

        // Transpose colMatrix before multiplication
        TensorType colMatrix_transposed = AdvancedTensorOperations<T>::transpose(colMatrix, 1, 0);

        // Perform matrix multiplication for weight gradients
        *weightGradients = AdvancedTensorOperations<T>::matmul(gradOutput_reshaped, colMatrix_transposed);
        *weightGradients = AdvancedTensorOperations<T>::reshape(*weightGradients, weightsShape);

        // Compute dX (input gradients)
        TensorType weightMatrix = AdvancedTensorOperations<T>::reshape(weightsTensor, {outputChannels, inputChannels * kernelHeight * kernelWidth});
        TensorType dX_col = AdvancedTensorOperations<T>::matmul(AdvancedTensorOperations<T>::transpose(weightMatrix, 1, 0), gradOutput_reshaped);
        gradInput = col2im(dX_col, inputShape, kernelHeight, kernelWidth, stride, padding, dilation);

        return gradInput;
    }

    void updateWeights(Optimizer<T>& optimizer) override {
        if (!weights || !biases || !weightGradients || !biasGradients) {
            throw std::runtime_error("Weights or gradients are not initialized!");
        }

        optimizer.optimize({std::ref(*weights), std::ref(*biases)},
                           {std::ref(*weightGradients), std::ref(*biasGradients)});
    }

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
        weights.emplace(Shape({outputChannels, inputChannels, kernelHeight, kernelWidth}));
        for (T& w : weights->getData()) w = static_cast<T>(RandomEngine::getHeInit(inputChannels * kernelHeight * kernelWidth));
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

        const std::vector<int>& inputStrides = input.getShape().getStride();

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

        const std::vector<int>& outputStrides = output.getShape().getStride();

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
