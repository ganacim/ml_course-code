add_executable(matrix_multiplication
    main.cpp
    kernel.cu kernel.h
    matrix.cpp matrix.h
    cpu.cpp cpu.h
    timer.cpp timer.h
    error.cpp error.h
)

target_link_libraries(matrix_multiplication PRIVATE nvtx3-cpp)

# if using add_library
# target_compile_features(template PUBLIC cxx_std_11)
# set_target_properties(template PROPERTIES CUDA_SEPARABLE_COMPILATION ON)