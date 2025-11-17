#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/TensorCreationUtil.hpp"

namespace sdnn {

class MNISTLoader {
public:
    MNISTLoader(const std::string& imagesPath, const std::string& labelsPath, int batchSize = 1, int numSamples = -1) 
        : imagesPath(imagesPath), labelsPath(labelsPath), batchSize(batchSize), numSamples(numSamples) {}

    std::pair<std::vector<Tensor>, std::vector<Tensor>> loadData() {
        std::vector<Tensor> images = loadImages();
        std::vector<Tensor> labels = loadLabels();
        return {images, labels};
    }

    std::string toAsciiArt(Tensor input) {
        std::ostringstream oss;
        const size_t height = 28;
        const size_t width = 28;
        const char* asciiChars = " .:-=+*#%@";  // 10 levels of intensity
        const size_t numLevels = 10;

        // If it's a 4D tensor (e.g., batch of images), we'll just show the first image
        std::vector<size_t> baseIndices(input.shape().rank() - 2, 0);

        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                std::vector<size_t> indices = baseIndices;
                indices.push_back(i);
                indices.push_back(j);
                float value = input.at<float>(indices);
                
                // Map the value to an ASCII character
                int index = std::min(static_cast<int>(value * numLevels), static_cast<int>(numLevels - 1));
                oss << asciiChars[index];
            }
            oss << "\n";
        }

        return oss.str();
    }

private:
    std::string imagesPath;
    std::string labelsPath;
    int batchSize;
    int numSamples;

    std::vector<Tensor> loadImages() {
        std::ifstream file(imagesPath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + imagesPath);
        }

        int32_t magicNumber = readInt(file);
        int32_t numImages = readInt(file);
        int32_t numRows = readInt(file);
        int32_t numCols = readInt(file);

        if (numSamples != -1) {
            numImages = numSamples;
        }

        if (magicNumber != 2051) {
            throw std::runtime_error("Invalid magic number in MNIST image file!");
        }
        
        std::vector<Tensor> images;
        for (int i = 0; i < numImages; i += batchSize) {
            int currentBatchSize = std::min(batchSize, numImages - i);

            Tensor image({currentBatchSize, 1, numRows, numCols}, 0.0f); 
            for (int j = 0; j < currentBatchSize; ++j) {
                for (int k = 0; k < numRows; ++k) {
                    for (int l = 0; l < numCols; ++l) {
                        unsigned char pixel = file.get();
                        image.set({static_cast<size_t>(j), 0UL, 
                                   static_cast<size_t>(k), static_cast<size_t>(l)}, 
                                  pixel / 255.0f);

                        float pixelValue = image.at<float>({static_cast<size_t>(j), 0UL, 
                                                            static_cast<size_t>(k), static_cast<size_t>(l)});
                        if (pixelValue < 0 || pixelValue > 1) {
                            throw std::runtime_error("Pixel value out of range: " + std::to_string(pixelValue));
                        }
                    }
                }
            }
            images.push_back(image);
        }
        return images;
    }

    std::vector<Tensor> loadLabels() {
        std::ifstream file(labelsPath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + labelsPath);
        }

        int32_t magicNumber = readInt(file);
        int32_t numLabels = readInt(file);

        if (numSamples != -1) {
            numLabels = numSamples;
        }

        if (magicNumber != 2049) {
            throw std::runtime_error("Invalid magic number in MNIST label file!");
        }

        std::vector<Tensor> labels;
        for (int i = 0; i < numLabels; i += batchSize) {
            int currentBatchSize = std::min(batchSize, numLabels - i);

            Tensor label(Shape({currentBatchSize, 10}), 0); 
            for (int j = 0; j < currentBatchSize; ++j) {
                int digit = file.get();
                label.set({static_cast<size_t>(j), static_cast<size_t>(digit)}, 1);
            }
            labels.push_back(label);
        }

        return labels;
    }

    int32_t readInt(std::ifstream& file) {
        unsigned char buffer[4];
        file.read(reinterpret_cast<char*>(buffer), 4);
        return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
    }
};
} // namespace sdnn

#endif // MNIST_LOADER_HPP
