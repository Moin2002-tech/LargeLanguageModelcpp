cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)

# Configure doctest
if(MSVC)
    add_compile_options(/utf-8)
endif()

project(LargeLanguageModelcpp)


#set(CMAKE_TOOLCHAIN_FILE "/home/moinshaikh/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Add tiktoken as a subdirectory
add_subdirectory(external/tiktoken)

find_package(fmt CONFIG REQUIRED)
find_package(pcre2 CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)




set(CMAKE_TOOLCHAIN_FILE "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain")
add_definitions(-DMODELS_DIR="${CMAKE_SOURCE_DIR}/datasets")
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
        include/Gpt2Untrained/Tests/memoryEstimateMHA_GQA.cpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/include/MultiHeadLatentV1.hpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/src/MultiHeadLatentV1.cpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/include/TransformBlockV5.hpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/src/TransformerBlockV5.cpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/include/gpt2mhl.hpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/src/gpt2mhl.cpp
        include/Gpt2Untrained/Tests/MultiHeadLatent/mhl.cpp)

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


        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch.lib"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch_cpu.lib"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/c10.lib"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/fmt.lib"

)

target_link_libraries(LargeLanguageModelcpp PRIVATE tiktoken

PCRE2::8BIT CURL::libcurl

)

set_target_properties(LargeLanguageModelcpp PROPERTIES
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/external/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        MAP_IMPORTED_CONFIG_DEBUG "DEBUG;RELEASE"
)

# Copy required DLLs next to the executable (RPATH does not work on Windows)
if(WIN32)
    add_custom_command(TARGET LargeLanguageModelcpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch_cpu.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/c10.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch_global_deps.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/uv.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/shm.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libiomp5md.dll"
            "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libiompstubs5md.dll"
            "$<TARGET_FILE_DIR:LargeLanguageModelcpp>"
    )
endif()
