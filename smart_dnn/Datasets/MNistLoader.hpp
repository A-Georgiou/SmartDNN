#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "smart_dnn/tensor/Tensor.hpp"

namespace smart_dnn {

template <typename T=float>
class MNISTLoader {
    using TensorType = Tensor<T>;
public:
    MNISTLoader(const std::string& imagesPath, const std::string& labelsPath, int batchSize = 1, int numSamples = -1) 
        : imagesPath(imagesPath), labelsPath(labelsPath), batchSize(batchSize), numSamples(numSamples) {}

    std::pair<std::vector<TensorType>, std::vector<TensorType>> loadData() {
        std::vector<TensorType> images = loadImages();
        std::vector<TensorType> labels = loadLabels();
        return {images, labels};
    }

    std::string toAsciiArt(Tensor<T> input) {
        std::ostringstream oss;
        const int height = 28;
        const int width = 28;
        const char* asciiChars = " .:-=+*#%@";  // 10 levels of intensity
        const int numLevels = 10;

        // If it's a 4D tensor (e.g., batch of images), we'll just show the first image
        std::vector<int> baseIndices(input.getShape().rank() - 2, 0);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                std::vector<int> indices = baseIndices;
                indices.push_back(i);
                indices.push_back(j);
                float value = input.at(indices);
                
                // Map the value to an ASCII character
                int index = std::min(static_cast<int>(value * numLevels), numLevels - 1);
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

    std::vector<TensorType> loadImages() {
        std::ifstream file(imagesPath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + imagesPath);
        }

        int magicNumber = readInt(file);
        int numImages = readInt(file);
        int numRows = readInt(file);
        int numCols = readInt(file);

        if (numSamples != -1) {
            numImages = numSamples;
        }

        if (magicNumber != 2051) {
            throw std::runtime_error("Invalid magic number in MNIST image file!");
        }
        
        std::vector<TensorType> images;
        for (int i = 0; i < numImages; i += batchSize) {
            int currentBatchSize = std::min(batchSize, numImages - i); // Handle last batch size

            TensorType image({currentBatchSize, 1, numRows, numCols}); 
            for (int j = 0; j < currentBatchSize; ++j) { // Read the actual batch size
                for (int k = 0; k < numRows; ++k) {
                    for (int l = 0; l < numCols; ++l) {
                        unsigned char pixel = file.get();
                        image.at({j, 0, k, l}) = pixel / T(255); // Normalize pixel values
                        if (image.at({j, 0, k, l}) < 0 || image.at({j, 0, k, l}) > 1) {
                            throw std::runtime_error("Pixel value out of range: " + std::to_string(image.at({j, 0, k, l})));
                        }
                    }
                }
            }
            images.push_back(image);
        }
        return images;
    }

    std::vector<TensorType> loadLabels() {
        std::ifstream file(labelsPath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + labelsPath);
        }

        int magicNumber = readInt(file);
        int numLabels = readInt(file);

        if (numSamples != -1) {
            numLabels = numSamples;
        }

        if (magicNumber != 2049) {
            throw std::runtime_error("Invalid magic number in MNIST label file!");
        }

        std::vector<TensorType> labels;
        for (int i = 0; i < numLabels; i += batchSize) {
            int currentBatchSize = std::min(batchSize, numLabels - i); // Handle last batch size

            TensorType label({currentBatchSize, 10}, T(0)); 
            for (int j = 0; j < currentBatchSize; ++j) { 
                unsigned char digit = file.get();
                label.at({j, digit}) = T(1);
            }
            labels.push_back(label);
        }

        return labels;
    }

    int readInt(std::ifstream& file) {
        unsigned char buffer[4];
        file.read(reinterpret_cast<char*>(buffer), 4);
        return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
    }
};

} // namespace smart_dnn

#endif // MNIST_LOADER_HPP
