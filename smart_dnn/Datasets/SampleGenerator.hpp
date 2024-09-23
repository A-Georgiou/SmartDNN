#ifndef SAMPLE_GENERATOR_HPP
#define SAMPLE_GENERATOR_HPP

#include <iostream>
#include <vector>
#include <random>
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/TensorCreationUtil.hpp"


namespace sdnn {

std::pair<std::vector<Tensor>, std::vector<Tensor>> generateLinearDataset(int num_samples, float noise=1.0) {
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> input_dist(0, 100.0);
    std::normal_distribution<> noise_dist(0, noise); 

    for (int i = 0; i < num_samples; ++i) {
        float x = input_dist(gen);
        float y = 2.0f * x + 3.0f + noise_dist(gen);

        Tensor input = Tensor(Shape{1}, x, dtype::f32); 
        Tensor target = Tensor(Shape{1}, y, dtype::f32);
        inputs.push_back(input);
        targets.push_back(target);
    }

    return {inputs, targets};
}

std::pair<std::vector<Tensor>, std::vector<Tensor>> generateBatchedLinearDataset(int num_samples, int batch_size, float noise=1.0) {
    std::vector<Tensor> input_batches;
    std::vector<Tensor> target_batches;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> input_dist(0, 100.0);
    std::normal_distribution<> noise_dist(0, noise);

    int num_batches = (num_samples + batch_size - 1) / batch_size;  // Ceiling division

    for (int batch = 0; batch < num_batches; ++batch) {
        int current_batch_size = std::min(batch_size, num_samples - batch * batch_size);

        Tensor input_batch(Shape{current_batch_size, 1}, 0, dtype::f32);
        Tensor target_batch(Shape{current_batch_size, 1}, 0, dtype::f32);

        for (int i = 0; i < current_batch_size; ++i) {
            float x = input_dist(gen);
            float y = 2.0f * x + 3.0f + noise_dist(gen);

            std::vector<size_t> indices = {static_cast<size_t>(i), 0};

            input_batch.set(indices, x);
            target_batch.set(indices, y);
        }

        input_batches.push_back(input_batch);
        target_batches.push_back(target_batch);
    }

    return {input_batches, target_batches};
}

} // namespace sdnn

#endif // SAMPLE_GENERATOR_HPP