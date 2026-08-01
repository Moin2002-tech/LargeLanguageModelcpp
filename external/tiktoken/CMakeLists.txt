cmake_minimum_required(VERSION 3.20)
project(tiktoken_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Aggressive optimization flags for maximum performance
if(CMAKE_BUILD_TYPE MATCHES "Release")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-O3 -march=native -mtune=native -ffast-math -funroll-loops)
    add_compile_options(-flto)
    add_link_options(-flto)
  endif()
endif()

add_library(tiktoken INTERFACE)
target_include_directories(tiktoken INTERFACE src)

set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
# Help CMake locate Conan CMakeDeps files if the toolchain didn't wire the path
list(APPEND CMAKE_PREFIX_PATH
  "${CMAKE_BINARY_DIR}/generators"
  "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/generators"
)

#find_package(PCRE2 REQUIRED CONFIG)
# 1. Handle PCRE2 via PkgConfig
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED libpcre2-8)

if(NOT TARGET PCRE2::8BIT)
    add_library(PCRE2::8BIT INTERFACE IMPORTED)
    target_include_directories(PCRE2::8BIT INTERFACE ${PCRE2_INCLUDE_DIRS})
    target_link_libraries(PCRE2::8BIT INTERFACE ${PCRE2_LIBRARIES})
endif()
set(_PCRE2_TARGET PCRE2::8BIT)

# 2. Handle utf8cpp manually (Assuming libutfcpp-dev is installed)
if(NOT TARGET utf8cpp::utf8cpp)
    add_library(utf8cpp::utf8cpp INTERFACE IMPORTED)
    # Ubuntu path for libutfcpp-dev
    target_include_directories(utf8cpp::utf8cpp INTERFACE /usr/include/utf8cpp)
endif()
set(_UTF8CPP_TARGET utf8cpp::utf8cpp)

# Now the rest of your script will work:
target_link_libraries(tiktoken INTERFACE ${_PCRE2_TARGET} ${_UTF8CPP_TARGET})
# utf8cpp can expose different target names depending on the package/provider
set(_UTF8CPP_TARGET "")
if (TARGET utf8::cpp)
  set(_UTF8CPP_TARGET utf8::cpp)
elseif (TARGET utf8cpp)
  set(_UTF8CPP_TARGET utf8cpp)
elseif (TARGET utf8cpp::utf8cpp)
  # Some distros/exporters may provide this namespace
  set(_UTF8CPP_TARGET utf8cpp::utf8cpp)
else()
  message(FATAL_ERROR "Could not detect a utf8cpp CMake target")
endif()

target_link_libraries(tiktoken INTERFACE ${_PCRE2_TARGET} ${_UTF8CPP_TARGET})

# Examples
option(TIKTOKEN_BUILD_EXAMPLES "Build tiktoken.cpp examples" ON)
option(TIKTOKEN_BUILD_BENCHMARKS "Build tiktoken.cpp benchmarks" ON)
if (TIKTOKEN_BUILD_EXAMPLES)
  add_executable(cpp_basic examples/cpp_basic.cpp)
  target_link_libraries(cpp_basic PRIVATE tiktoken)
  target_compile_features(cpp_basic PRIVATE cxx_std_17)
  target_include_directories(cpp_basic PRIVATE src)
endif()

if (TIKTOKEN_BUILD_BENCHMARKS)
  find_package(Threads REQUIRED)
  
  add_executable(tiktoken_benchmark benchmarks/encode_benchmark.cpp)
  target_link_libraries(tiktoken_benchmark PRIVATE tiktoken)
  target_compile_features(tiktoken_benchmark PRIVATE cxx_std_17)
  target_include_directories(tiktoken_benchmark PRIVATE src)

  add_executable(benchmark benchmarks/benchmark.cpp)
  target_link_libraries(benchmark PRIVATE tiktoken Threads::Threads)
  target_compile_features(benchmark PRIVATE cxx_std_17)
  target_include_directories(benchmark PRIVATE src)
endif()
