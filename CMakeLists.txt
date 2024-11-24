cmake_minimum_required(VERSION 3.22)

project(brute LANGUAGES C CXX CUDA)

set(CMAKE_CXX_FLAGS "-Ofast -march=native -funroll-loops -fstrict-aliasing -fno-stack-protector -flto")

set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED) 

find_package(nlohmann_json 3.2.0 REQUIRED) 

include_directories(include /usr/local/cuda/include)

add_executable(brute ./src/main.cu)

target_link_libraries(brute nlohmann_json::nlohmann_json)

# target_link_libraries(brute 
#     /usr/local/lib/libsecp256k1.a
#     OpenSSL::Crypto 
#     ${CURL_LIBRARIES}
#     /usr/lib/x86_64-linux-gnu/libcurlpp.a
# )
