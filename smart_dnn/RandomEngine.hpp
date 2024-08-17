#ifndef RANDOMENGINE_H
#define RANDOMENGINE_H

#include <random>

class RandomEngine {
public:
    static std::mt19937& getInstance() {
        static std::random_device rd;
        static std::mt19937 engine(rd());
        return engine;
    }

    static float getRandRange(float min, float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(getInstance());
    }

    static float getRand() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(getInstance());
    }

private:
    RandomEngine() = default;
    ~RandomEngine() = default;

    RandomEngine(const RandomEngine&) = delete;
    RandomEngine& operator=(const RandomEngine&) = delete;
};

#endif // RANDOMENGINE_H