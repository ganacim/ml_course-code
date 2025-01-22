#include <iostream>
#include <unistd.h>
#include <cstring>

#include "kernel.h"
#include "matrix.h"
#include "cpu.h"

using namespace std;

void help(const char *name){
    cout << "Usage: " << name << " " << endl;
    cout << "    --size:<int>    Size of the matrix (default: 1024)" << endl;
}

int match(const char *argument, const char *option, const char **value) {
    int length = (int) strlen(option);
    if (option[length-1] == ':') length--;
    if (strncmp(argument, option, length) == 0 && 
        (argument[length] == ':' || argument[length] == '(' 
            || !argument[length])) {
        if (value) {
            if (option[length] == ':' && argument[length] == ':') length++;
            *value = argument+length;
        }
        return 1;
    }
    else return 0;
}

int main(int argc, const char* argv[]) {
    cout << "Matrix Multiplication Example." << endl;
    // size of the matrix (assuming square matrix)
    int matrix_size = 1024;
    bool do_print_matrix = false;
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
        if (match(argv[i], "--size:", &value)){
            matrix_size = atoi(value);
        }
        else if (strcmp(argv[i], "--print") == 0){
            do_print_matrix = true;
        }
    }
    // print running parameters
    cout << "Matrix size: " << matrix_size << endl;
    // cout << "Running on CPU: " << (run_cpu ? "yes" : "no") << endl;
    // Creata a matrix of size matrix_size x matrix_size
    vector<float> matrix = create_random_matrix(matrix_size);
    if (do_print_matrix){
        print_matrix(matrix, matrix_size);
    }

    return 0;
}