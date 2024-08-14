#include <vector>
#include <iostream>

class Tensor {
public:
    Tensor();
    Tensor(int rows, int cols);
    Tensor(int rows, int cols, float value);
    Tensor(const std::vector<std::vector<float>>& data);
    Tensor(const Tensor& other);
    ~Tensor();

    Tensor& operator=(const Tensor& other);
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    void fill(float value);
    void randomize(float min, float max);
    void print() const;

    int rows() const;
    int cols() const;
    float& operator()(int i, int j);
    const float& operator()(int i, int j) const;

    void save(std::ostream& os) const;
    void load(std::istream& is);

private:
    std::vector<int> shape;
    std::vector<float> data;
    float* d_data; // Pointer to GPU memory
    bool onGPU;

    void allocateGPUMemory();
    void freeGPUMemory();
    void copyToGPU();
    void copyToCPU();
};