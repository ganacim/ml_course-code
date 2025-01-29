#include "cuda.h"
#include "timer.h"

#include <stdio.h>

using namespace std;

// Define a kernel function, which is the entry point
// for execution on the GPU
__global__ void matrix_multiplication(float *m1, float *m2, float *result, unsigned int m1_rows, unsigned int m1_cols, unsigned int m2_cols)
{
    // Get the row and column of the current element
    unsigned int i = blockIdx.x;
    unsigned int j = blockIdx.y;

    // Compute the dot product of the row of m1 and the column of m2
    float value = 0;
    for (unsigned int k = 0; k < m1_cols; k++) {
        value += m1[i * m1_cols + k] * m2[k * m2_cols + j];
    }

    // Store the result in the output matrix
    result[i * m2_cols + j] = value;
}

vector<float> cuda_multiplication(const std::vector<float>& m1,
                                    const std::vector<float>& m2,
                                    unsigned int m1_rows,
                                    unsigned int m1_cols,
                                    unsigned int m2_cols)
{
    auto& timer = util::timers.gpu_add("CUDA Multiplication");
    // Allocate memory on the host
    vector<float> result(m1_rows * m2_cols);
    // Allocate memory on the device
    float *d_m1, *d_m2, *d_result;
    cudaMalloc(&d_m1, m1_rows * m1_cols * sizeof(float));
    cudaMalloc(&d_m2, m1_cols * m2_cols * sizeof(float));
    cudaMalloc(&d_result, m1_rows * m2_cols * sizeof(float));
    // Copy data from host to device
    cudaMemcpy(d_m1, m1.data(), m1_rows * m1_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, m2.data(), m1_cols * m2_cols * sizeof(float), cudaMemcpyHostToDevice);
    // Define grid and block size
    dim3 grid(m1_rows, m2_cols);
    dim3 block(1, 1);
    // Launch kernel
    matrix_multiplication<<<grid, block>>>(d_m1, d_m2, d_result, m1_rows, m1_cols, m2_cols);
    // Copy data from device to host
    cudaMemcpy(result.data(), d_result, m1_rows * m2_cols * sizeof(float), cudaMemcpyDeviceToHost);
    // Free memory on the device
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_result);
    timer.stop();
    return result;
}