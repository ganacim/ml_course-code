#include <iostream>
#include <unistd.h>
#include <cstring>
#include <regex>

#include "kernel.h"
#include "matrix.h"
#include "cpu.h"
#include "timer.h"

using namespace std;

void help(const char *name){
    cout << "Usage: " << name << " " << endl;
    cout << "    --size:<m1_rows>:<m1_cols>:<m1_cols> or --size:<matrix_size>" << endl;
    cout << "           example: --size:1024:1024:1024 or --size:1024" << endl;
    cout << "    --print -- prints all matrices." << endl;
    cout << "    --save:<output_file> -- saves matrices M1 and M2 to file." << endl;
    cout << "    --load:<input_file> -- load matrices M1 and M2 from file." << endl;
    cout << "    --cpu[:<output_file>] -- run CPU version and save result to file (optional)." << endl;
}

int main(int argc, const char* argv[]) {
    auto& timer = util::timers.cpu_add("Total time");
    cout << "Matrix Multiplication Example." << endl;
    // Variables for holding parameters
    // matrix size
    regex const msize1_re{R"~(--size:(\d+))~"}; 
    regex const msize3_re{R"~(--size:(\d+):(\d+):(\d+))~"}; 
    // https://stackoverflow.com/questions/56710024/what-is-a-raw-string
    // size of the matrices
    unsigned int m1_rows = 1024, m1_cols = 1024, m2_cols = 1024;
    // save and load
    bool do_save_matrix = false;
    bool do_load_matrix = false;
    // does not work with full paths e.g.: "../output.txt"
    // but works with "folder/output.txt"
    regex const save_re{R"~(--save:([\w\-_\/]+\.?\w*))~"};
    string output_file_name = "output.txt";
    regex const load_re{R"~(--load:([\w\-_\/]+\.?\w*))~"};
    string input_file_name = "input.txt";
    // running options
    bool do_run_cpu = true;
    regex const cpu_re{R"~(--cpu:([\w\-_\/]+\.?\w*))~"};
    bool do_save_cpu_result = false;
    string cpu_result_file = "cpu_result.txt";
    // other options
    bool do_print_matrix = false;
    // Matrices
    vector<float> m1, m2;
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
        // --save:output_matrix_file
        else if (strncmp(argv[i], "--save:", 7) == 0){
            if (regex_match(arg, m, save_re)){
                do_save_matrix = true;
                output_file_name = m[1].str();
            }
            else {
                cout << "Invalid output file name: '" << argv[i] + 7 << "'" << endl;
                exit(1);
            }
        }
        else if (strncmp(argv[i], "--load:", 7) == 0){
            if (regex_match(arg, m, load_re)){
                do_load_matrix = true;
                input_file_name = m[1].str();
            }
            else {
                cout << "Invalid input file name: '" << argv[i] + 7 << "'" << endl;
                exit(1);
            }
        }
        else if (strncmp(argv[i], "--cpu:", 5) == 0){
            do_run_cpu = true;
            if (regex_match(arg, m, cpu_re)){
                do_save_cpu_result = true;
                cpu_result_file = m[1].str();
            }
        }
    }
    if (do_load_matrix){
        cout << "Loading matrices from file: " << input_file_name << endl;
        load_matrices(input_file_name, m1, m2, m1_rows, m1_cols, m2_cols);
    }
    else {
        cout << "Creating random matrices." << endl;
        // Creata a matrix of size matrix_size x matrix_size
        m1 = create_random_matrix(m1_rows, m1_cols);
        m2 = create_random_matrix(m1_cols, m2_cols);
    }
    // print running parameters
    cout << "Matrix size: M1["<< m1_rows << ", " << m1_cols << "] M2[" << m1_cols << ", " << m2_cols << "]" << endl;
    if (do_print_matrix){
        cout << "Matrix 1:" << endl;
        print_matrix(m1, m1_rows);
        cout << "Matrix 2:" << endl;
        print_matrix(m2, m1_cols);
    }
    if (do_save_matrix){
        cout << "Saving matrices to file: " << output_file_name << endl;
        save_matrices(output_file_name, m1, m2, m1_rows, m1_cols, m2_cols);
    }
    // Run the CPU version
    cout << "Running on CPU: " << (do_run_cpu ? "yes" : "no") << endl;
    if (do_run_cpu) {
        vector<float> result_cpu = cpu_naive_multiplication(m1, m2, m1_rows, m1_cols, m2_cols);
        if (do_save_cpu_result){
            cout << "Saving CPU result to file: " << cpu_result_file << endl;
            save_matrix(cpu_result_file, result_cpu, m1_rows, m2_cols);
        }
        if (do_print_matrix){
            cout << endl << "CPU result:" << endl;
            print_matrix(result_cpu, m1_rows);
        }
    }
    timer.stop();
    util::timers.flush();
    return 0;
}