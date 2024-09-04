#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "../Tensor/Tensor.hpp"

namespace smart_dnn {

template <typename T=float>
class MNISTLoader {
    using TensorType = Tensor<T>;
public:
    MNISTLoader(const std::string& imagesPath, const std::string& labelsPath, int batchSize = 1) 
        : imagesPath(imagesPath), labelsPath(labelsPath), batchSize(batchSize) {}

    std::pair<std::vector<TensorType>, std::vector<TensorType>> loadData() {
        std::vector<TensorType> images = loadImages();
        std::vector<TensorType> labels = loadLabels();
        return {images, labels};
    }

private:
    std::string imagesPath;
    std::string labelsPath;
    int batchSize;

    std::vector<TensorType> loadImages() {
        std::ifstream file(imagesPath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + imagesPath);
        }

        int magicNumber = readInt(file);
        int numImages = readInt(file);
        int numRows = readInt(file);
        int numCols = readInt(file);

        if (magicNumber != 2051) {
            throw std::runtime_error("Invalid magic number in MNIST image file!");
        }
        
        std::vector<TensorType> images;
        for (int i = 0; i < numImages; i += batchSize) {
            TensorType image({batchSize, 1, numRows, numCols}); 
            for (int j = 0; j < batchSize; ++j) { // Read batchSize images at a time
                for (int k = 0; k < numRows; ++k) {
                    for (int l = 0; l < numCols; ++l) {
                        unsigned char pixel = file.get();
                        image.at({j, 0, k, l}) = pixel / T(255); // Normalize pixel values
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

        if (magicNumber != 2049) {
            throw std::runtime_error("Invalid magic number in MNIST label file!");
        }

        std::vector<TensorType> labels;
        
        for (int i = 0; i < numLabels; i += batchSize) {
            TensorType label({batchSize, 10}, T(0));
            for (int j = 0; j < batchSize; ++j) { 
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
