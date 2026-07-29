cmake_minimum_required(VERSION 3.25)

project(ptcg_cuda_engine LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures to build")

option(PTCG_CUDA_BUILD_BENCHMARK "Build the standalone CUDA smoke benchmark" ON)
option(PTCG_CUDA_BUILD_TORCH "Build the optional PyTorch extension" OFF)

add_library(ptcg_cuda_engine STATIC
    src/engine_kernels.cu
)
target_include_directories(ptcg_cuda_engine PUBLIC include)
target_link_libraries(ptcg_cuda_engine PUBLIC CUDA::cudart)
target_compile_options(ptcg_cuda_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
)
set_target_properties(ptcg_cuda_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

if(PTCG_CUDA_BUILD_BENCHMARK)
    add_executable(ptcg_cuda_smoke benchmarks/cuda_smoke.cu)
    target_link_libraries(ptcg_cuda_smoke PRIVATE ptcg_cuda_engine)

    add_executable(ptcg_cuda_official_setup benchmarks/official_setup.cu)
    target_include_directories(ptcg_cuda_official_setup PRIVATE include)
    target_link_libraries(ptcg_cuda_official_setup PRIVATE CUDA::cudart)

    add_executable(ptcg_cuda_status_core benchmarks/official_status_core.cu)
    target_include_directories(ptcg_cuda_status_core PRIVATE include)
    target_link_libraries(ptcg_cuda_status_core PRIVATE CUDA::cudart)
endif()

if(PTCG_CUDA_BUILD_TORCH)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    find_package(Torch REQUIRED)
    find_library(
        TORCH_PYTHON_LIBRARY
        NAMES torch_python
        PATHS "${Torch_DIR}/../../../lib"
        NO_DEFAULT_PATH
        REQUIRED
    )
    add_library(_ptcg_cuda MODULE src/torch_binding.cpp)
    target_include_directories(_ptcg_cuda PRIVATE include)
    target_link_libraries(
        _ptcg_cuda
        PRIVATE
            ptcg_cuda_engine
            "${TORCH_LIBRARIES}"
            "${TORCH_PYTHON_LIBRARY}"
            Python3::Module
    )
    separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(_ptcg_cuda PRIVATE ${TORCH_CXX_FLAGS_LIST})
    target_compile_definitions(_ptcg_cuda PRIVATE TORCH_EXTENSION_NAME=_ptcg_cuda)
    set_target_properties(_ptcg_cuda PROPERTIES PREFIX "")
endif()
