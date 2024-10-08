#include <gtest/gtest.h>

#ifdef USE_ARRAYFIRE_TENSORS
    #include "arrayfire.h"
#endif

int main(int argc, char **argv) {
    #ifdef USE_ARRAYFIRE_TENSORS
        af::setBackend(AF_BACKEND_CPU);
    #endif

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}