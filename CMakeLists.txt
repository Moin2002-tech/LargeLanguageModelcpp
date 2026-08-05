cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(TORCH_CUDA_ARCH_LIST "7.5")
set(CAFFE2_USE_CUDNN 9)

project(LargeLanguageModelcpp)


set(CMAKE_TOOLCHAIN_FILE "/home/moinshaikh/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Add tiktoken as a subdirectory
add_subdirectory(external/tiktoken)

find_package(fmt CONFIG REQUIRED)
find_package(pcre2 CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)





add_executable(LargeLanguageModelcpp src/dataPreparationAndSampling.cpp
        include/basics/Text.h
        src/basics/Text.cpp
        include/basics/tokenize.h
        src/basics/tokenize.cpp
        include/basics/SimpleTokenizerV1.h
        src/basics/SimpleTokenizerV1.cpp
        include/basics/GPTDatasetV1.h
        src/basics/GPTDatasetV1.cpp
        external/third_party/doctest.hpp
        external/third_party/doctest.hpp
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
        include/Gpt2Untrained/Tests/memoryEstimateMHA_GQA.cpp)

target_include_directories(LargeLanguageModelcpp PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/external/libtorch/include"
        "${CMAKE_SOURCE_DIR}/external/libtorch/include/torch/csrc/api/include"
        "${CMAKE_SOURCE_DIR}/external/third_party"


)
target_include_directories(LargeLanguageModelcpp PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
)



# Link against tiktoken, pcre2, and curl
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

        # 2. CUDA Driver (The one usually missing - found in system libs)
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1"

        # 3. NVRTC (Compiler library)
        "/usr/local/cuda/lib64/libnvrtc.so"

        fmt::fmt
        # System base libs
        pthread
        dl
        m
)

target_link_libraries(LargeLanguageModelcpp PRIVATE tiktoken

PCRE2::8BIT CURL::libcurl

)

set_target_properties(LargeLanguageModelcpp PROPERTIES
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/external/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        MAP_IMPORTED_CONFIG_DEBUG "DEBUG;RELEASE"
)