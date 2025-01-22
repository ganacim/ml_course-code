# add_library(kernels
#     kernel.cu kernel.cuh
# )
add_executable(main
    main.cpp
    kernel.cu kernel.h
)

# target_compile_features(kernels PUBLIC cxx_std_11)
# set_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_link_libraries(main kernels)