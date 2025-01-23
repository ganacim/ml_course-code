#include <iostream>
#include <unistd.h>
#include <cstring>
#include <regex>

#include "kernel.h"
#include "matrix.h"
#include "cpu.h"

using namespace std;

void help(const char *name){
    cout << "Usage: " << name << " " << endl;
    cout << "    --size:<int>    Size of the matrix (default: 1024)" << endl;
}

int main(int argc, const char* argv[]) {
    cout << "Matrix Multiplication Example." << endl;
    // size of the matrix (assuming square matrix)
    // m2_rows = m1_cols !!!!
    unsigned int m1_rows = 1024, m1_cols = 1024, m2_cols = 1024;
    // https://stackoverflow.com/questions/56710024/what-is-a-raw-string
    std::regex const msize1_re{R"~(--size:(\d+))~"}; 
    std::regex const msize3_re{R"~(--size:(\d+):(\d+):(\d+))~"}; 
    bool do_print_matrix = false;
    bool do_save_matrix = false;
    string output_matrix_file = "output_matrix.txt";
    // Parse arguments
    if (argc <= 1){
        help(argv[0]);
        exit(1);
    }
    for (int i = 0; i < argc; i++){
        if (strcmp(argv[i], "-h") == 0 ||
                strcmp(argv[i], "--help") == 0){
            help(argv[0]);
            exit(0);
        }
    }
    for (int i = 0; i < argc; i++){
        const char *value;
        std::smatch m;
        const std::string arg(argv[i]);
        // --size:m1_rows:m1_cols:m2_cols
        if(regex_match(arg, m, msize3_re)) {
            if (m.size() == 4){
                m1_rows = std::stoi(m[1]);
                m1_cols = std::stoi(m[2]);
                m2_cols = std::stoi(m[3]);
            }
            else {
                cout << "Invalid matrix size: " << arg << endl;
                exit(1);
            }
        }
        // --size:matrix_size // for square matrix
        else if(regex_match(arg, m, msize1_re)) {
            if (m.size() == 2) {
                m1_rows = m1_cols = m2_cols = stoi(m[1]);
            }
            else {
                cout << "Invalid matrix size: " << arg << endl;
                exit(1);
            }
        }
        // --print
        else if (strcmp(argv[i], "--print") == 0){
            do_print_matrix = true;
        }
    }
    // print running parameters
    cout << "Matrix size: M1["<< m1_rows << ", " << m1_cols << "] M2[" << m1_cols << ", " << m2_cols << "]" << endl;
    // cout << "Running on CPU: " << (run_cpu ? "yes" : "no") << endl;
    // Creata a matrix of size matrix_size x matrix_size
    vector<float> m1 = create_random_matrix(m1_rows, m1_cols);
    vector<float> m2 = create_random_matrix(m1_cols, m2_cols);
    if (do_print_matrix){
        cout << "Matrix 1:" << endl;
        print_matrix(m1, m1_rows);
        cout << "Matrix 2:" << endl;
        print_matrix(m2, m1_cols);
    }
    // Run the CPU version
    vector<float> result_cpu = cpu_multiplication(m1, m2, m1_rows, m1_cols, m2_cols);
    if (do_print_matrix){
        cout << endl << "CPU result:" << endl;
        print_matrix(result_cpu, m1_rows);
    }
    return 0;
}