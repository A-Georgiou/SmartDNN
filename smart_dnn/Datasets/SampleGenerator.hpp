#include <iostream>
#include <vector>
#include <random>
#include "../Tensor.hpp"

std::pair<std::vector<Tensor>, std::vector<Tensor>> generateLinearDataset(int num_samples) {
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> input_dist(0, 100.0);
    std::normal_distribution<> noise_dist(0, 1.0); 

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