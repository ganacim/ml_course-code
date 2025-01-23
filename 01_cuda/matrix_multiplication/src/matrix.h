#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <vector>

std::vector<float> create_random_matrix(unsigned int rows, unsigned int cols);
void print_matrix(const std::vector<float> &matrix, unsigned int rows);

#endif