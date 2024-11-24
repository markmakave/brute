
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

__managed__ lumina::ecdsa::u256 x;


__global__ void kernel()
{
    x = x / 3;
}

int main()
{
    x = lumina::ecdsa::u256(11, 0, 0, 0);

    std::cout << x << '\n';
    kernel <<<1, 1>>> ();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << x << '\n';

    // static constexpr size_t N = 1024 * 1024;

    // std::vector<std::array<uint8_t, 128>> messages(N);

    // static std::mt19937 engine(42);
    // static std::uniform_int_distribution<uint8_t> dis('a', 'z');

    // for (auto& message : messages)
    // {
    //     for (size_t j = 0; j < 65; ++j)
    //         message[j] = dis(engine);

    //     //

    //     message[65] = 0x80;

    //     for (size_t j = 66; j < 120; ++j)
    //         message[j] = 0x00;

    //     uint64_t size_bits = 65 * 8;
    //     message[120] = (size_bits >> 56) & 0xFF;
    //     message[121] = (size_bits >> 48) & 0xFF;
    //     message[122] = (size_bits >> 40) & 0xFF;
    //     message[123] = (size_bits >> 32) & 0xFF;
    //     message[124] = (size_bits >> 24) & 0xFF;
    //     message[125] = (size_bits >> 16) & 0xFF;
    //     message[126] = (size_bits >> 8) & 0xFF;
    //     message[127] = size_bits & 0xFF;
    // }

    //

    // uint8_t* d_messages;
    // uint8_t* d_hashes;

    // CUDA_CHECK(cudaMalloc(&d_messages, messages.size() * sizeof(messages[0])));
    // CUDA_CHECK(cudaMalloc(&d_hashes,   N * (256 / 8)));

    // CUDA_CHECK(cudaMemcpy(d_messages, messages.data(), messages.size() * sizeof(messages[0]), cudaMemcpyHostToDevice));

    // size_t n = N;

    // dim3 blocks(N / 512 + 1), threads(512);
    // std::array<void*, 4> args = { &n, &d_messages, &d_hashes };

    // std::vector<std::array<uint8_t, 256 / 8>> hashes(N);

    // CUDA_CHECK(cudaLaunchKernel(btc::sha256<128>, blocks, threads, args.data(), 0, 0));
    // CUDA_CHECK(cudaDeviceSynchronize());

    // auto start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 1000; ++i)
    // {
    //     CUDA_CHECK(cudaMemcpy(d_messages, messages.data(), messages.size() * sizeof(messages[0]), cudaMemcpyHostToDevice));
    //     CUDA_CHECK(cudaLaunchKernel(btc::sha256<128>, blocks, threads, args.data(), 0, 0));
    //     CUDA_CHECK(cudaDeviceSynchronize());
    //     CUDA_CHECK(cudaMemcpy(hashes.data(), d_hashes, hashes.size() * sizeof(hashes[0]), cudaMemcpyDeviceToHost));
    // }
    // CUDA_CHECK(cudaDeviceSynchronize());
    // auto finish = std::chrono::high_resolution_clock::now();

    // std::cout << "Bandwidth:" << double(10000 * N) / std::chrono::duration_cast<std::chrono::seconds>(finish - start).count() << '\n';

    return 0;
}
