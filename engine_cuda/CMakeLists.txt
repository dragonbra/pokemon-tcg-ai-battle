cmake_minimum_required(VERSION 3.25)

project(ptcg_cuda_engine LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures to build")
set(
    PTCG_CUDA_OFFICIAL_THREADS
    "2"
    CACHE STRING
    "CUDA threads per block for one-thread-per-environment official kernels"
)
if(NOT PTCG_CUDA_OFFICIAL_THREADS MATCHES "^[0-9]+$")
    message(FATAL_ERROR "PTCG_CUDA_OFFICIAL_THREADS must be an integer in [1, 1024]")
endif()
if(
    PTCG_CUDA_OFFICIAL_THREADS LESS 1
    OR PTCG_CUDA_OFFICIAL_THREADS GREATER 1024
)
    message(FATAL_ERROR "PTCG_CUDA_OFFICIAL_THREADS must be an integer in [1, 1024]")
endif()

option(PTCG_CUDA_BUILD_BENCHMARK "Build the standalone CUDA smoke benchmark" ON)
option(PTCG_CUDA_BUILD_TORCH "Build the optional PyTorch extension" OFF)

add_library(ptcg_cuda_engine STATIC
    src/engine_kernels.cu
    src/official_engine_kernels.cu
)
target_include_directories(ptcg_cuda_engine PUBLIC include)
target_link_libraries(ptcg_cuda_engine PUBLIC CUDA::cudart)
target_compile_definitions(
    ptcg_cuda_engine
    PRIVATE PTCG_CUDA_OFFICIAL_THREADS=${PTCG_CUDA_OFFICIAL_THREADS}
)
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

    add_executable(ptcg_cuda_continuation_vm_smoke benchmarks/continuation_vm_smoke.cu)
    target_include_directories(ptcg_cuda_continuation_vm_smoke PRIVATE include)
    target_link_libraries(ptcg_cuda_continuation_vm_smoke PRIVATE CUDA::cudart)

    add_executable(
        ptcg_cuda_official_continuation_dispatch_smoke
        benchmarks/official_continuation_dispatch_smoke.cu
    )
    target_include_directories(
        ptcg_cuda_official_continuation_dispatch_smoke
        PRIVATE include
    )
    target_link_libraries(
        ptcg_cuda_official_continuation_dispatch_smoke
        PRIVATE CUDA::cudart
    )

    add_executable(ptcg_cuda_official_rule_pack_smoke benchmarks/official_rule_pack_smoke.cu)
    target_include_directories(ptcg_cuda_official_rule_pack_smoke PRIVATE include)
    target_link_libraries(ptcg_cuda_official_rule_pack_smoke PRIVATE CUDA::cudart)

    add_executable(ptcg_cuda_official_core_pod_smoke benchmarks/official_core_pod_smoke.cu)
    target_include_directories(ptcg_cuda_official_core_pod_smoke PRIVATE include)
    target_link_libraries(ptcg_cuda_official_core_pod_smoke PRIVATE CUDA::cudart)

    add_executable(
        ptcg_cuda_official_effect_interpreter_smoke
        benchmarks/official_effect_interpreter_smoke.cu
    )
    target_include_directories(ptcg_cuda_official_effect_interpreter_smoke PRIVATE include)
    target_link_libraries(
        ptcg_cuda_official_effect_interpreter_smoke
        PRIVATE CUDA::cudart
    )

    add_executable(
        ptcg_cuda_official_continual_refresh_smoke
        benchmarks/official_continual_refresh_smoke.cu
    )
    target_include_directories(
        ptcg_cuda_official_continual_refresh_smoke
        PRIVATE include
    )
    target_link_libraries(
        ptcg_cuda_official_continual_refresh_smoke
        PRIVATE CUDA::cudart
    )

    add_executable(
        ptcg_cuda_official_trigger_pull_smoke
        benchmarks/official_trigger_pull_smoke.cu
    )
    target_include_directories(
        ptcg_cuda_official_trigger_pull_smoke
        PRIVATE include
    )
    target_link_libraries(
        ptcg_cuda_official_trigger_pull_smoke
        PRIVATE CUDA::cudart
    )

    add_executable(
        ptcg_cuda_official_trigger_resolver_smoke
        benchmarks/official_trigger_resolver_smoke.cu
    )
    target_include_directories(
        ptcg_cuda_official_trigger_resolver_smoke
        PRIVATE include
    )
    target_link_libraries(
        ptcg_cuda_official_trigger_resolver_smoke
        PRIVATE CUDA::cudart
    )

    add_executable(
        ptcg_cuda_official_knockout_smoke
        benchmarks/official_knockout_smoke.cu
    )
    target_include_directories(
        ptcg_cuda_official_knockout_smoke
        PRIVATE include
    )
    target_link_libraries(
        ptcg_cuda_official_knockout_smoke
        PRIVATE CUDA::cudart
    )

    add_executable(
        ptcg_cuda_official_turn_flow_smoke
        src/official_engine_kernels.cu
        benchmarks/official_turn_flow_smoke.cu
    )
    target_include_directories(
        ptcg_cuda_official_turn_flow_smoke
        PRIVATE include
    )
    target_link_libraries(
        ptcg_cuda_official_turn_flow_smoke
        PRIVATE CUDA::cudart
    )

    add_executable(
        ptcg_cuda_official_attack_smoke
        benchmarks/official_attack_smoke.cu
    )
    target_include_directories(
        ptcg_cuda_official_attack_smoke
        PRIVATE include
    )
    target_link_libraries(
        ptcg_cuda_official_attack_smoke
        PRIVATE CUDA::cudart
    )

    add_executable(
        ptcg_cuda_official_runtime_smoke
        benchmarks/official_runtime_smoke.cu
    )
    target_include_directories(
        ptcg_cuda_official_runtime_smoke
        PRIVATE include
    )
    target_link_libraries(
        ptcg_cuda_official_runtime_smoke
        PRIVATE ptcg_cuda_engine CUDA::cudart
    )

    add_executable(
        ptcg_cuda_official_flow_resume_diagnostic
        benchmarks/official_flow_resume_diagnostic.cu
    )
    target_include_directories(
        ptcg_cuda_official_flow_resume_diagnostic
        PRIVATE include
    )
    target_link_libraries(
        ptcg_cuda_official_flow_resume_diagnostic
        PRIVATE CUDA::cudart
    )

    add_executable(
        ptcg_cuda_official_main_smoke
        benchmarks/official_main_smoke.cu
    )
    target_include_directories(
        ptcg_cuda_official_main_smoke
        PRIVATE include
    )
    target_link_libraries(
        ptcg_cuda_official_main_smoke
        PRIVATE ptcg_cuda_engine CUDA::cudart
    )
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
