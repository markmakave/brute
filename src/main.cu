
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <chrono>

#include "cuda/sha256.cuh"
#include "cuda/u256.cuh"
#include "cuda/ecdsa.cuh"

#define CUDA_CHECK(x) if (auto error = (x); error != cudaSuccess) std::cerr << "CUDA Error " __FILE__ ":" << __LINE__ << ": " << cudaGetErrorString(error) << '\n';

__global__ void kernel(
    size_t n,
    const lumina::ecdsa::u256* __restrict__ x,
    const lumina::ecdsa::u256* __restrict__ y,
          lumina::ecdsa::u256* __restrict__ d
) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    d[i] = x[i] / y[i];
}

lumina::ecdsa::u256 random_u256()
{
    lumina::ecdsa::u256 result;

    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_int_distribution<lumina::ecdsa::u64> dist;

    for (int i = 0; i < 4; ++i)
        result._u64[i] = dist(mt);

    return result;
}

int main(int argc, char** argv)
{
    constexpr size_t n = 1024 * 1024;

    std::vector<lumina::ecdsa::u256> x(n), y(n), d(n);

    for (size_t i = 0; i < n; ++i)
    {
        x[i] = random_u256();
        y[i] = random_u256();
    }

    lumina::ecdsa::u256 *d_x, *d_y, *d_d;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(*d_x) * n));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(*d_y) * n));
    CUDA_CHECK(cudaMalloc(&d_d, sizeof(*d_d) * n));

    CUDA_CHECK(cudaMemcpy(d_x, x.data(), sizeof(*d_x) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), sizeof(*d_y) * n, cudaMemcpyHostToDevice));

    size_t threads = 512, blocks = n / threads + 1;
    kernel <<<blocks, threads>>> (n, d_x, d_y, d_d);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto now = std::chrono::steady_clock::now();
    for (size_t i = 0; i < 1000; ++i)
    kernel <<<blocks, threads>>> (n, d_x, d_y, d_d);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto elapsed = std::chrono::steady_clock::now() - now;

    std::cout << "Bandwidth: " << (n * 1000) / (std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000000.0) << " B/s\n";

    return 0;
}
