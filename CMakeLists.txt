cmake_minimum_required(VERSION 3.22)

project(parser LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS "-O3 -fopenmp")

set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL REQUIRED)

add_executable(parser ./src/main.cpp)

include_directories(include)
target_link_libraries(parser "/usr/local/lib/libsecp256k1.a" OpenSSL::Crypto)
