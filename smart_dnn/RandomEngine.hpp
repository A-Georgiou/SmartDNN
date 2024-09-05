#ifndef RANDOMENGINE_HPP
#define RANDOMENGINE_HPP

#include <random>

class RandomEngine {
public:
    static std::mt19937& getInstance() {
        static std::mt19937 engine;
        return engine;
    }

    static void setSeed(unsigned int seed) {
        getInstance().seed(seed);
    }

    static float getRandRange(float min, float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(getInstance());
    }

    static float getRand() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(getInstance());
    }

    static float getXavierInit(float n_in) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(1.0f / n_in));
        return dist(getInstance());
    }

    static float getHeInit(float n_in) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / n_in));
        return dist(getInstance());
    }

    static float getHeRandRange(float n_in, float min, float max) {
        std::normal_distribution<float> he_dist(0.0f, std::sqrt(2.0f / n_in));
        float he_value = he_dist(getInstance());

        return std::min(std::max(he_value, min), max);
    }

private:
    RandomEngine() = default;
    ~RandomEngine() = default;

    RandomEngine(const RandomEngine&) = delete;
    RandomEngine& operator=(const RandomEngine&) = delete;
};

#endif // RANDOMENGINE_HPP