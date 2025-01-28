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

int match(const string& arg, const char *pattern, smatch &m){
    const regex re_str{string(pattern)};
    regex_match(arg, m, re_str);
    return m.size();
}

int main(int argc, const char* argv[]) {
    auto& timer = util::timers.cpu_add("Total time");
    cout << "# Matrix Multiplication." << endl;
    // Variables for holding parameters
    // size of the matrices
    unsigned int m1_rows = 1024, m1_cols = 1024, m2_cols = 1024;
    // save and load
    bool save_matrix_flag = false;
    bool load_matrix_flag = false;
    string output_file_name = "output.txt";
    string input_file_name = "input.txt";
    // running options
    bool run_cpu_flag = false;
    bool save_cpu_result_flag = false;
    string cpu_result_file = "cpu_result.txt";
    // other options
    bool print_matrix_flag = false;
    // Matrices
    vector<float> m1, m2, cpu_result;
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
    for (int i = 1; i < argc; i++){
        const char *value;
        std::smatch m;
        const std::string arg(argv[i]);
        // --size:m1_rows // m1_rows = m1_cols = m2_cols
        if (match(arg, R"~(--size:(\d+))~", m) == 2){
            // https://stackoverflow.com/questions/56710024/what-is-a-raw-string
            m1_rows = m1_cols = m2_cols = std::stoi(m[1]);
        }
        // --size:m1_rows:m1_cols:m2_cols
        else if (match(arg, R"~(--size:(\d+):(\d+):(\d+))~", m) == 4){
            m1_rows = std::stoi(m[1]);
            m1_cols = std::stoi(m[2]);
            m2_cols = std::stoi(m[3]);
        }
        // --print
        else if (match(argv[i], "--print", m)){
            print_matrix_flag = true;
        }
        // --save:output_matrix_file
        else if (match(argv[i], R"~(--save:([\w\-_\/]+(?:\.\w+)?))~", m)){
            save_matrix_flag = true;
            output_file_name = m[1].str();
        }
        // --load:input_matrix_file
        else if (match(argv[i], R"~(--load:([\w\-_\/]+(?:\.\w+)?))~", m)){
            load_matrix_flag = true;
            input_file_name = m[1].str();
        }
        // --cpu[:output_file]
        else if (match(arg, R"~(--cpu(?::([\w\-_\/]+(?:\.\w+)?))?)~", m)){
            run_cpu_flag = true;
            if (m.size() == 2 and m[1].str().size() > 0){
                save_cpu_result_flag = true;
                cpu_result_file = m[1].str();
            }
        }
        // Invalid argument
        else {
            cout << "! Invalid argument: " << argv[i] << endl;
            exit(1);
        }
    }
    if (load_matrix_flag){
        cout << "> Loading matrices from file: " << input_file_name << endl;
        load_matrices(input_file_name, m1, m2, m1_rows, m1_cols, m2_cols);
    }
    else {
        cout << "> Creating random matrices." << endl;
        // Creata a matrix of size matrix_size x matrix_size
        m1 = create_random_matrix(m1_rows, m1_cols);
        m2 = create_random_matrix(m1_cols, m2_cols);
    }
    // print running parameters
    cout << ">> size: M1["<< m1_rows << ", " << m1_cols << "] M2[" << m1_cols << ", " << m2_cols << "]" << endl;
    if (save_matrix_flag){
        cout << "> Saving matrices to file: " << output_file_name << endl;
        save_matrices(output_file_name, m1, m2, m1_rows, m1_cols, m2_cols);
    }
    // Run the CPU version
    cout << "> Running on CPU: " << (run_cpu_flag ? "yes" : "no") << endl;
    if (run_cpu_flag) {
        cpu_result = cpu_naive_multiplication(m1, m2, m1_rows, m1_cols, m2_cols);
        if (save_cpu_result_flag){
            cout << ">> Saving CPU result to file: " << cpu_result_file << endl;
            save_matrix(cpu_result_file, cpu_result, m1_rows, m2_cols);
        }
    }
    if (print_matrix_flag){
        cout << "> Printing Matrices" << endl;
        cout << ">> Matrix 1:" << endl;
        print_matrix(m1, m1_rows);
        cout << ">> Matrix 2:" << endl;
        print_matrix(m2, m1_cols);
        if (run_cpu_flag){
            cout << endl << ">> CPU result:" << endl;
            print_matrix(cpu_result, m1_rows);
        }
    }
    timer.stop();
    util::timers.flush();
    return 0;
}