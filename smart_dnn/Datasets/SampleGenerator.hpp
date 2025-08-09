#ifndef SAMPLE_GENERATOR_HPP
#define SAMPLE_GENERATOR_HPP

#include <iostream>
#include <vector>
#include <random>
#include "smart_dnn/Tensor/Tensor.hpp"

namespace smart_dnn {

std::pair<std::vector<Tensor<float>>, std::vector<Tensor<float>>> generateLinearDataset(int num_samples, float noise=1.0) {
    std::vector<Tensor<float>> inputs;
    std::vector<Tensor<float>> targets;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> input_dist(0, 100.0);
    std::normal_distribution<> noise_dist(0, noise); 

    for (int i = 0; i < num_samples; ++i) {
        float x = input_dist(gen);
        float y = 2.0f * x + 3.0f + noise_dist(gen);

        Tensor input(Shape{1}, x); 
        Tensor target(Shape{1}, y);
        inputs.push_back(input);
        targets.push_back(target);
    }

    return {inputs, targets};
}

std::pair<std::vector<Tensor<float>>, std::vector<Tensor<float>>> generateBatchedLinearDataset(int num_samples, int batch_size, float noise=1.0) {
    std::vector<Tensor<float>> input_batches;
    std::vector<Tensor<float>> target_batches;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> input_dist(0, 100.0);
    std::normal_distribution<> noise_dist(0, noise);

    int num_batches = (num_samples + batch_size - 1) / batch_size;  // Ceiling division

    for (int batch = 0; batch < num_batches; ++batch) {
        int current_batch_size = std::min(batch_size, num_samples - batch * batch_size);

        Tensor<float> input_batch(Shape{current_batch_size, 1});
        Tensor<float> target_batch(Shape{current_batch_size, 1});

        for (int i = 0; i < current_batch_size; ++i) {
            float x = input_dist(gen);
            float y = 2.0f * x + 3.0f + noise_dist(gen);

            input_batch.at({i, 0}) = x;
            target_batch.at({i, 0}) = y;
        }

        input_batches.push_back(input_batch);
        target_batches.push_back(target_batch);
    }

    return {input_batches, target_batches};
}

} // namespace smart_dnn

#endif // SAMPLE_GENERATOR_HPP