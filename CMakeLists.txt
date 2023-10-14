cmake_minimum_required(VERSION 3.22)

project(parser LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS "-O3")

set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED) 

add_executable(parser ./src/main.cpp)

include_directories(include lib/json/single_include)
target_link_libraries(parser 
    /usr/local/lib/libsecp256k1.a
    OpenSSL::Crypto 
    ${CURL_LIBRARIES}
    # /usr/lib/aarch64-linux-gnu/libcurlpp.a
    /usr/lib/x86_64-linux-gnu/libcurlpp.a
)
