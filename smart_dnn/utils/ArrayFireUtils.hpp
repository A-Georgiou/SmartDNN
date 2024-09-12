#include "smart_dnn/shape/Shape.hpp"
#include <vector>
#include <arrayfire.h>

namespace sdnn::af_utils {

static af::array indexAfArray(const af::array& input, const std::vector<int>& indices, const Shape& shape) {
    if (indices.size() != shape.rank()) {
        throw std::invalid_argument("Number of indices must match the tensor's dimensions.");
    }

    std::vector<af_seq> af_indices(indices.size());

    // Convert the indices into ArrayFire sequences (af_seq)
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= static_cast<int>(shape[i])) {
            throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
        }
        af_indices[i] = af_make_seq(indices[i], indices[i], 1);  // Create an af_seq for the specific index
    }

    // Create an empty output array to store the result
    af_array out_array;

    // Perform N-dimensional indexing using af_index
    af_err err = af_index(&out_array, input.get(), static_cast<unsigned>(indices.size()), af_indices.data());

    if (err != AF_SUCCESS) {
        throw std::runtime_error("ArrayFire index operation failed.");
    }

    // Return the result as an af::array
    return af::array(out_array);
}

}