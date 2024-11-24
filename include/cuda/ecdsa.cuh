#pragma once

#include "cuda/u256.cuh"

namespace lumina::ecdsa
{

// __constant__ static u256_modulo GX = { 0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07, 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798 };
// __constant__ static u256_modulo GY = { 0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8, 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8 };

__constant__ static u64 P[4]  = { 0xFFFFFC2FFFFFFFFE, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };

static u256 inverse_mod(const u256& x, const u256& m)
{
    // uint64_t t = 0, new_t = 1;
    // uint64_t r = p, new_r = a;

    // while (new_r != 0) {
    //     uint64_t quotient = r / new_r;

    //     // Update t and new_t
    //     uint64_t temp_t = t;
    //     t = new_t;
    //     new_t = temp_t - quotient * new_t;

    //     // Update r and new_r
    //     uint64_t temp_r = r;
    //     r = new_r;
    //     new_r = temp_r - quotient * new_r;
    // }

    // if (r > 1) {
    //     throw std::invalid_argument("Modular inverse does not exist");
    // }

    // // Ensure the result is positive
    // if (t > p) {
    //     t = t % p;
    // } else if (t < 0) {
    //     t += p;
    // }

    // return t;
}

__device__
static u256 div_mod(const u256& lhs, const u256& rhs, const u256& m)
{
    return lhs * inverse_mod(rhs, m) % m;
}

struct point
{
    u256 x, y;

    __device__
    point operator+ (const point& rhs) const
    {
        u256 slope = div_mod((y - rhs.y), (x - rhs.x), P);
        u256 x_r = slope * slope - x - rhs.x;
        u256 y_r = y + slope * (x_r - x);

        return { x_r % P, -y_r % P };
    }

    __device__
    point operator* (const u256& scalar)
    {

    }

};

}
