#include "cpu.h"
#include "timer.h"

using namespace std;

// Naive matrix multiplication
vector<float> cpu_multiplication(const std::vector<float>& m1,
                                    const std::vector<float>& m2,
                                    unsigned int m1_rows,
                                    unsigned int m1_cols,
                                    unsigned int m2_cols)
{
    auto& timer = util::timers.cpu_add("CPU Multiplication");
    vector<float> result(m1_rows * m2_cols);
    for (int i = 0; i < m1_rows; i++) {
        for (int j = 0; j < m2_cols; j++) {
            float sum = 0.0;
            for (int k = 0; k < m1_cols; k++) {
                sum += m1[i * m1_cols + k] * m2[k * m2_cols + j];
            }
            result[i * m2_cols + j] = sum;
        }
    }
    timer.stop();
    return result;
}