#include "cuda.h"

#include <stdio.h>

using namespace std;

// Define a device function, which 
// can be called from a kernel and executes on the GPU
__device__ int device_function(){
    printf("Hello CUDA World!\n");
    return 1;
}

// Define a kernel function, which is the entry point
// for execution on the GPU
__global__ void kernel() {
    device_function();
}

// Define a wrapper function, which launches the kernel
void kernel_wrapper() {
    // Launch kernel with <<<block, thread>>> syntax
    kernel<<<1,32>>>();
}

vector<float> cuda_multiplication(const std::vector<float>& m1,
                                    const std::vector<float>& m2,
                                    unsigned int m1_rows,
                                    unsigned int m1_cols,
                                    unsigned int m2_cols)
{
}