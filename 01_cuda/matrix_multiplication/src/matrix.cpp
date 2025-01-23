#include "matrix.h"

#include <iostream>
#include <ctime>
#include <random>

typedef std::mt19937 RNG;  // Mersenne Twister with a popular choice of parameters

std::vector<float> create_random_matrix(unsigned int rows, unsigned int cols)
{
    // Create a normal distribution with mean 0 and standard deviation 1
    uint32_t seed = (uint32_t) time(0);    
    RNG rng(seed);
    std::normal_distribution<float> normal(0.0, 1.0);
    // Create a matrix of size matrix_size x matrix_size with random values
    std::vector<float> matrix(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = normal(rng);
        }
    }
    return matrix;
}

void print_matrix(const std::vector<float> &matrix, unsigned int rows)
{
    unsigned int cols = matrix.size() / rows;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}