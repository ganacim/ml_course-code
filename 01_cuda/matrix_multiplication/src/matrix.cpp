#include "matrix.h"

#include <iostream>
#include <ctime>
#include <random>

typedef std::mt19937 RNG;  // Mersenne Twister with a popular choice of parameters

std::vector<float> create_random_matrix(int matrix_size)
{
    // Create a normal distribution with mean 0 and standard deviation 1
    uint32_t seed = (uint32_t) time(0);    
    RNG rng(seed);
    std::normal_distribution<float> normal(0.0, 1.0);
    // Create a matrix of size matrix_size x matrix_size with random values
    std::vector<float> matrix(matrix_size * matrix_size);
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            matrix[i * matrix_size + j] = normal(rng);
        }
    }
    return matrix;
}

void print_matrix(const std::vector<float> &matrix, int matrix_size)
{
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            std::cout << matrix[i * matrix_size + j] << " ";
        }
        std::cout << std::endl;
    }
}