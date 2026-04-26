#ifndef DATA_AUGMENTATION_LAYER_HPP
#define DATA_AUGMENTATION_LAYER_HPP

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include "smart_dnn/Layer.hpp"
#include "smart_dnn/RandomEngine.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"

namespace smart_dnn {

template <typename T=float>
class RandomFlip2DLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    RandomFlip2DLayer(bool horizontal = true, bool vertical = false, float probability = 0.5f)
        : horizontal(horizontal), vertical(vertical), probability(probability) {
        if (probability < 0.0f || probability > 1.0f) {
            throw std::invalid_argument("RandomFlip2DLayer: probability must be between 0 and 1");
        }
    }

    TensorType forward(const TensorType& input) override {
        validateImageTensor(input, "RandomFlip2DLayer");

        lastHorizontalFlips.assign(input.getShape()[0], false);
        lastVerticalFlips.assign(input.getShape()[0], false);

        if (!this->trainingMode) {
            return input;
        }

        TensorType output(input.getShape());
        int batchSize = input.getShape()[0];
        int channels = input.getShape()[1];
        int height = input.getShape()[2];
        int width = input.getShape()[3];

        for (int n = 0; n < batchSize; ++n) {
            lastHorizontalFlips[n] = horizontal && shouldApply();
            lastVerticalFlips[n] = vertical && shouldApply();

            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    int sourceH = lastVerticalFlips[n] ? height - 1 - h : h;
                    for (int w = 0; w < width; ++w) {
                        int sourceW = lastHorizontalFlips[n] ? width - 1 - w : w;
                        output.at({n, c, h, w}) = input.at({n, c, sourceH, sourceW});
                    }
                }
            }
        }

        return output;
    }

    TensorType backward(const TensorType& gradOutput) override {
        validateImageTensor(gradOutput, "RandomFlip2DLayer");
        if (lastHorizontalFlips.size() != static_cast<size_t>(gradOutput.getShape()[0]) ||
            lastVerticalFlips.size() != static_cast<size_t>(gradOutput.getShape()[0])) {
            throw std::runtime_error("RandomFlip2DLayer: forward pass must be called before backward");
        }

        TensorType gradInput(gradOutput.getShape());
        int batchSize = gradOutput.getShape()[0];
        int channels = gradOutput.getShape()[1];
        int height = gradOutput.getShape()[2];
        int width = gradOutput.getShape()[3];

        for (int n = 0; n < batchSize; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    int sourceH = lastVerticalFlips[n] ? height - 1 - h : h;
                    for (int w = 0; w < width; ++w) {
                        int sourceW = lastHorizontalFlips[n] ? width - 1 - w : w;
                        gradInput.at({n, c, h, w}) = gradOutput.at({n, c, sourceH, sourceW});
                    }
                }
            }
        }

        return gradInput;
    }

private:
    bool horizontal;
    bool vertical;
    float probability;
    std::vector<bool> lastHorizontalFlips;
    std::vector<bool> lastVerticalFlips;

    bool shouldApply() const {
        return RandomEngine::getRand() < probability;
    }

    static void validateImageTensor(const TensorType& tensor, const char* layerName) {
        if (tensor.getShape().rank() != 4) {
            throw std::invalid_argument(std::string(layerName) + ": input tensor must have rank 4 (batch, channels, height, width)");
        }
    }
};

template <typename T=float>
class RandomRotation90Layer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    explicit RandomRotation90Layer(int turns = -1) : configuredTurns(turns) {
        if (turns < -1 || turns > 3) {
            throw std::invalid_argument("RandomRotation90Layer: turns must be -1 or between 0 and 3");
        }
    }

    TensorType forward(const TensorType& input) override {
        validateImageTensor(input, "RandomRotation90Layer");

        lastInputShape = input.getShape();
        lastTurns = this->trainingMode ? selectTurns() : 0;
        hasForwardPass = true;
        return rotate(input, lastTurns);
    }

    TensorType backward(const TensorType& gradOutput) override {
        validateImageTensor(gradOutput, "RandomRotation90Layer");
        if (!hasForwardPass) {
            throw std::runtime_error("RandomRotation90Layer: forward pass must be called before backward");
        }

        TensorType gradInput = rotate(gradOutput, (4 - lastTurns) % 4);
        if (gradInput.getShape() != lastInputShape) {
            throw std::runtime_error("RandomRotation90Layer: gradient shape does not match previous input shape");
        }
        return gradInput;
    }

private:
    int configuredTurns;
    int lastTurns = 0;
    Shape lastInputShape{{1, 1, 1, 1}};
    bool hasForwardPass = false;

    int selectTurns() const {
        if (configuredTurns >= 0) {
            return configuredTurns;
        }
        return std::min(3, static_cast<int>(RandomEngine::getRand() * 4.0f));
    }

    static TensorType rotate(const TensorType& input, int turns) {
        int normalizedTurns = ((turns % 4) + 4) % 4;
        int batchSize = input.getShape()[0];
        int channels = input.getShape()[1];
        int inputHeight = input.getShape()[2];
        int inputWidth = input.getShape()[3];
        bool swapsDimensions = normalizedTurns == 1 || normalizedTurns == 3;
        int outputHeight = swapsDimensions ? inputWidth : inputHeight;
        int outputWidth = swapsDimensions ? inputHeight : inputWidth;

        TensorType output({batchSize, channels, outputHeight, outputWidth});

        for (int n = 0; n < batchSize; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < outputHeight; ++h) {
                    for (int w = 0; w < outputWidth; ++w) {
                        int sourceH = h;
                        int sourceW = w;
                        if (normalizedTurns == 1) {
                            sourceH = inputHeight - 1 - w;
                            sourceW = h;
                        } else if (normalizedTurns == 2) {
                            sourceH = inputHeight - 1 - h;
                            sourceW = inputWidth - 1 - w;
                        } else if (normalizedTurns == 3) {
                            sourceH = w;
                            sourceW = inputWidth - 1 - h;
                        }
                        output.at({n, c, h, w}) = input.at({n, c, sourceH, sourceW});
                    }
                }
            }
        }

        return output;
    }

    static void validateImageTensor(const TensorType& tensor, const char* layerName) {
        if (tensor.getShape().rank() != 4) {
            throw std::invalid_argument(std::string(layerName) + ": input tensor must have rank 4 (batch, channels, height, width)");
        }
    }
};

template <typename T=float>
class RandomCrop2DLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    RandomCrop2DLayer(int cropHeight, int cropWidth, int top = -1, int left = -1)
        : cropHeight(cropHeight), cropWidth(cropWidth), configuredTop(top), configuredLeft(left) {
        if (cropHeight <= 0 || cropWidth <= 0) {
            throw std::invalid_argument("RandomCrop2DLayer: crop dimensions must be positive");
        }
        if ((top < 0) != (left < 0)) {
            throw std::invalid_argument("RandomCrop2DLayer: top and left must both be set or both be random");
        }
    }

    TensorType forward(const TensorType& input) override {
        validateImageTensor(input, "RandomCrop2DLayer");
        if (cropHeight > input.getShape()[2] || cropWidth > input.getShape()[3]) {
            throw std::invalid_argument("RandomCrop2DLayer: crop dimensions cannot exceed input dimensions");
        }

        lastInputShape = input.getShape();
        int batchSize = input.getShape()[0];
        int channels = input.getShape()[1];
        int inputHeight = input.getShape()[2];
        int inputWidth = input.getShape()[3];

        selectCropOrigin(inputHeight, inputWidth);
        hasForwardPass = true;
        TensorType output({batchSize, channels, cropHeight, cropWidth});

        for (int n = 0; n < batchSize; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < cropHeight; ++h) {
                    for (int w = 0; w < cropWidth; ++w) {
                        output.at({n, c, h, w}) = input.at({n, c, lastTop + h, lastLeft + w});
                    }
                }
            }
        }

        return output;
    }

    TensorType backward(const TensorType& gradOutput) override {
        validateImageTensor(gradOutput, "RandomCrop2DLayer");
        if (!hasForwardPass) {
            throw std::runtime_error("RandomCrop2DLayer: forward pass must be called before backward");
        }
        if (gradOutput.getShape()[2] != cropHeight || gradOutput.getShape()[3] != cropWidth) {
            throw std::invalid_argument("RandomCrop2DLayer: gradient output shape must match crop dimensions");
        }
        if (gradOutput.getShape()[0] != lastInputShape[0] || gradOutput.getShape()[1] != lastInputShape[1]) {
            throw std::invalid_argument("RandomCrop2DLayer: gradient output batch and channel dimensions must match input");
        }

        TensorType gradInput(lastInputShape, T(0));
        int batchSize = gradOutput.getShape()[0];
        int channels = gradOutput.getShape()[1];

        for (int n = 0; n < batchSize; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < cropHeight; ++h) {
                    for (int w = 0; w < cropWidth; ++w) {
                        gradInput.at({n, c, lastTop + h, lastLeft + w}) = gradOutput.at({n, c, h, w});
                    }
                }
            }
        }

        return gradInput;
    }

private:
    int cropHeight;
    int cropWidth;
    int configuredTop;
    int configuredLeft;
    int lastTop = 0;
    int lastLeft = 0;
    Shape lastInputShape{{1, 1, 1, 1}};
    bool hasForwardPass = false;

    void selectCropOrigin(int inputHeight, int inputWidth) {
        int maxTop = inputHeight - cropHeight;
        int maxLeft = inputWidth - cropWidth;

        if (configuredTop >= 0) {
            if (configuredTop > maxTop || configuredLeft > maxLeft) {
                throw std::invalid_argument("RandomCrop2DLayer: configured crop origin is outside input dimensions");
            }
            lastTop = configuredTop;
            lastLeft = configuredLeft;
        } else if (this->trainingMode) {
            lastTop = randomIntInclusive(maxTop);
            lastLeft = randomIntInclusive(maxLeft);
        } else {
            lastTop = maxTop / 2;
            lastLeft = maxLeft / 2;
        }
    }

    static int randomIntInclusive(int maxValue) {
        if (maxValue <= 0) {
            return 0;
        }
        return std::min(maxValue, static_cast<int>(RandomEngine::getRand() * static_cast<float>(maxValue + 1)));
    }

    static void validateImageTensor(const TensorType& tensor, const char* layerName) {
        if (tensor.getShape().rank() != 4) {
            throw std::invalid_argument(std::string(layerName) + ": input tensor must have rank 4 (batch, channels, height, width)");
        }
    }
};

} // namespace smart_dnn

#endif // DATA_AUGMENTATION_LAYER_HPP
