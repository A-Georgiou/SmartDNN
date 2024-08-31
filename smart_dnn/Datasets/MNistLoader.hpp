#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "../Tensor/Tensor.hpp"

class MNISTLoader {
public:
    MNISTLoader(const std::string& imagesPath, const std::string& labelsPath, int batchSize = 1) 
        : imagesPath(imagesPath), labelsPath(labelsPath), batchSize(batchSize) {}

    std::pair<std::vector<Tensor>, std::vector<Tensor>> loadData() {
        std::vector<Tensor> images = loadImages();
        std::vector<Tensor> labels = loadLabels();
        return {images, labels};
    }

private:
    std::string imagesPath;
    std::string labelsPath;
    int batchSize;

    std::vector<Tensor> loadImages() {
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
        
        std::vector<Tensor> images;
        for (int i = 0; i < numImages; i += batchSize) {
            Tensor image({batchSize, 1, numRows, numCols}); 
            for (int j = 0; j < batchSize; ++j) { // Read batchSize images at a time
                for (int k = 0; k < numRows; ++k) {
                    for (int l = 0; l < numCols; ++l) {
                        unsigned char pixel = file.get();
                        image({j, 0, k, l}) = pixel / 255.0f; // Normalize pixel values
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

        int magicNumber = readInt(file);
        int numLabels = readInt(file);

        if (magicNumber != 2049) {
            throw std::runtime_error("Invalid magic number in MNIST label file!");
        }

        std::vector<Tensor> labels;
        
        for (int i = 0; i < numLabels; i += batchSize) {
            Tensor label({batchSize, 10}, 0.0f);
            for (int j = 0; j < batchSize; ++j) { 
                unsigned char digit = file.get();
                label({j, digit}) = 1.0f;
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

#endif // MNIST_LOADER_HPP
