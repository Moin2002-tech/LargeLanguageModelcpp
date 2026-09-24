cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TORCH_CUDA_ARCH_LIST "7.5")


# Platform-specific setup (toolchain, CUDA, compiler options)

if(MSVC)
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
    set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}")
    set(CMAKE_TOOLCHAIN_FILE "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain")
    if(MSVC)
        add_compile_options(/utf-8)
    endif()
else()
    set(CAFFE2_USE_CUDNN 9)
    set(CMAKE_TOOLCHAIN_FILE "/home/moinshaikh/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
endif()

project(LargeLanguageModelcpp)

# Third-party packages must be located *before* the tiktoken subproject runs:
# the CONFIG packages define the authoritative imported targets
# (pcre2::pcre2-8-shared, CURL::libcurl_shared, PCRE2::8BIT, ...) with absolute
# library paths. tiktoken's `if(NOT TARGET PCRE2::8BIT)` guard then reuses the
# vcpkg target instead of building its own from the bare pkg-config name, which
# the MSVC linker cannot resolve (LNK1181: cannot open input file 'pcre2-8.lib').
find_package(fmt CONFIG REQUIRED)
find_package(pcre2 CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)

# Add tiktoken as a subdirectory
add_subdirectory(external/tiktoken)

if(WIN32)
    # 1. Define the CUDA toolkit root path FIRST
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")

    # 2. Call find_package using the modern CUDAToolkit package name
    find_package(CUDAToolkit REQUIRED)
endif()

add_definitions(-DDATASETS_DIR="${CMAKE_SOURCE_DIR}/datasets/")
add_definitions(-DMODELS_DIR="${CMAKE_SOURCE_DIR}/datasets")


# Sources (superset of both platform lists — remove any that don't exist
# yet on a given machine)

add_executable(LargeLanguageModelcpp src/dataPreparationAndSampling.cpp
        include/basics/Text.h
        src/basics/Text.cpp
        include/basics/tokenize.h
        src/basics/tokenize.cpp
        include/basics/SimpleTokenizerV1.h
        src/basics/SimpleTokenizerV1.cpp
        include/basics/GPTDatasetV1.h
        src/basics/GPTDatasetV1.cpp
        external/third_party/doctest.cpp
        src/attentionMechanism.cpp
        include/AttentionMechanism/AttentionMechanism.hpp
        src/AttentionMechanism/AttentionMechanism.cpp
        include/AttentionMechanism/CausalAttentionMechanism.hpp
        src/AttentionMechanism/CausalAttention.cpp
        include/AttentionMechanism/MultiHeadAttention.hpp
        src/AttentionMechanism/MultiHeadAttention.cpp
        include/GPT2LargeLanguageModel/Gpt2Model.hpp
        include/GPT2LargeLanguageModel/util.hpp
        src/GPT2LargeLanguageModel/Gpt2Model.cpp
        include/GPT2LargeLanguageModel/TransformerBlock.hpp
        src/GPT2LargeLanguageModel/TransformerBlock.cpp
        include/Gpt2Untrained/dataPreparation.hpp
        src/Gpt2Untrained/dataPreparation.cpp
        include/Gpt2Untrained/Tests.cpp
        include/Gpt2Untrained/Tests/multiParameterModel.hpp
        include/Gpt2Untrained/Tests/multiDropout.cpp
        include/Gpt2Untrained/Tests/FlopsAnalysis.cpp
        include/Gpt2Untrained/Tests/kv_cache/include/MultiHeadAttentionV2.hpp
        include/Gpt2Untrained/Tests/kv_cache/include/LayerNormV2.hpp
        include/Gpt2Untrained/Tests/kv_cache/src/MultiHeadAttentionV2.cpp
        include/Gpt2Untrained/Tests/kv_cache/src/LayerNormV2.cpp
        include/Gpt2Untrained/Tests/kv_cache/include/GELU.hpp
        include/Gpt2Untrained/Tests/kv_cache/src/GELU.cpp
        include/Gpt2Untrained/Tests/kv_cache/include/FeedForward.hpp
        include/Gpt2Untrained/Tests/kv_cache/src/FeedForward.cpp
        include/Gpt2Untrained/Tests/kv_cache/include/TransformBlockV2.hpp
        include/Gpt2Untrained/Tests/kv_cache/src/TransformBlockV2.cpp
        include/Gpt2Untrained/Tests/kv_cache/include/GptModelV2.hpp
        include/Gpt2Untrained/Tests/kv_cache/src/GptModelV2.cpp
        include/Gpt2Untrained/Tests/kv_cache/kv_cache.cpp
        include/Gpt2Untrained/Tests/GroupQueryAttention/include/GroupQueryAttentionV1.hpp
        include/Gpt2Untrained/Tests/GroupQueryAttention/src/GroupQueryAttentionV1.cpp
        include/Gpt2Untrained/Tests/GroupQueryAttention/include/LayerNormV3.hpp
        include/Gpt2Untrained/Tests/GroupQueryAttention/src/LayerNormV3.cpp
        include/Gpt2Untrained/Tests/GroupQueryAttention/include/GELUV3.hpp
        include/Gpt2Untrained/Tests/GroupQueryAttention/include/FeedForwardV3.hpp
        include/Gpt2Untrained/Tests/GroupQueryAttention/src/FeedForwardV3.cpp
        include/Gpt2Untrained/Tests/GroupQueryAttention/include/TransformerBlockV4.hpp
        include/Gpt2Untrained/Tests/GroupQueryAttention/src/TransformerBlockV4.cpp
        include/Gpt2Untrained/Tests/GroupQueryAttention/include/Gpt2ModelV3.hpp
        include/Gpt2Untrained/Tests/GroupQueryAttention/src/Gpt2ModelV3.cpp
        include/Gpt2Untrained/Tests/GroupQueryAttention/GQA.cpp
        include/Gpt2Untrained/Tests/memoryEstimateMHA_GQA.cpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/include/MultiHeadLatentV1.hpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/src/MultiHeadLatentV1.cpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/include/TransformBlockV5.hpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/src/TransformerBlockV5.cpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/include/gpt2mhl.hpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/src/gpt2mhl.cpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/mhl.cpp
        src/TrainedGpt/preTrainingOnunlabled.cpp
        src/TrainedGpt/WeightLoading/WeightLoading.cpp
        include/TrainedGpt/generateText.hpp
        include/TrainedGpt/TrainingGuttenbergDatasets/prepairDatasets.hpp
        src/TrainedGpt/TrainingGuttenbergDatasets/prepairDatasets.cpp
        src/TrainedGpt/TrainingGuttenbergDatasets/tests.cpp
        src/TrainedGpt/TrainingGuttenbergDatasets/combineTextfiles.cpp
)

target_include_directories(LargeLanguageModelcpp PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/external/libtorch/include"
        "${CMAKE_SOURCE_DIR}/external/libtorch/include/torch/csrc/api/include"
        "${CMAKE_SOURCE_DIR}/external/third_party"
)

# ---------------------------------------------------------------------------
# CUDA toolkit include directories
#
# The bundled LibTorch is a CUDA build (2.11.0+cu128), so the CUDA-aware
# headers that some translation units pull in (c10/cuda/CUDACachingAllocator.h,
# c10/cuda/CUDAFunctions.h -> c10/cuda/CUDAException.h ->
# c10/cuda/CUDAMiscFunctions.h) end with `#include <cuda_runtime.h>`, which
# lives in the CUDA toolkit, not in LibTorch.
#
# NOTE: find_package(CUDAToolkit) exports CUDAToolkit_INCLUDE_DIRS.
# The legacy `CUDA_INCLUDE_DIRS` variable belongs to the old FindCUDA module,
# which no longer exists in modern CMake, so using it silently added nothing
# to the include path and cl.exe failed with
#   fatal error C1083: Cannot open include file: 'cuda_runtime.h'
# ---------------------------------------------------------------------------
if(WIN32)
    if(NOT CUDAToolkit_INCLUDE_DIRS)
        message(FATAL_ERROR
                "find_package(CUDAToolkit) did not report any include directories. "
                "Set CUDAToolkit_ROOT to your CUDA toolkit installation, e.g. "
                "-DCUDAToolkit_ROOT=\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8\"")
    endif()
    target_include_directories(LargeLanguageModelcpp PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()


# ---------------------------------------------------------------------------
# Linking (platform-specific LibTorch/CUDA libraries)
# ---------------------------------------------------------------------------
if(WIN32)
    target_link_libraries(LargeLanguageModelcpp PRIVATE
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch.lib"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch_cpu.lib"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch_cuda.lib"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/c10_cuda.lib"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/c10.lib"
    )
else()
    target_link_libraries(LargeLanguageModelcpp PRIVATE
            -Wl,--no-as-needed
            -Wl,--whole-archive
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch_cuda.so"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch_cpu.so"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch.so"
            -Wl,--no-whole-archive

            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libc10_cuda.so"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libc10.so"

            # 1. CUDA Runtime (from your toolkit)
            "/usr/local/cuda/lib64/libcudart.so"

            # 2. CUDA Driver (usually missing - found in system libs)
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1"

            # 3. NVRTC (Compiler library)
            "/usr/local/cuda/lib64/libnvrtc.so"

            fmt::fmt
            pthread
            dl
            m
    )
endif()

# Required for linking CUDA symbols into LibTorch on MSVC


target_link_libraries(LargeLanguageModelcpp PRIVATE
        tiktoken
        PCRE2::8BIT
        CURL::libcurl
)

set_target_properties(LargeLanguageModelcpp PROPERTIES
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/external/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        MAP_IMPORTED_CONFIG_DEBUG "DEBUG;RELEASE"
)


if(WIN32)
    target_link_options(LargeLanguageModelcpp PRIVATE
            "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()
# Copy required DLLs next to the executable (RPATH does not work on Windows)
if(WIN32)
    file(GLOB TORCH_DLLS "${CMAKE_SOURCE_DIR}/external/libtorch/lib/*.dll")
    add_custom_command(TARGET LargeLanguageModelcpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            "$<TARGET_FILE_DIR:LargeLanguageModelcpp>"
    )
endif()