#pragma once

#include "cuda/u256.cuh"

namespace lumina::ecdsa
{

// __constant__ static u256 GX = { 0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07, 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798 };
// __constant__ static u256 GY = { 0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8, 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8 };

static constexpr u64 P_raw[] = { 0xFFFFFC2FFFFFFFFE, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
__constant__ static u256 P;

__device__ __forceinline__
static u256 inverse_mod(const u256& x, const u256& m)
{
    u256 t = 0, new_t = 1;
    u256 r = m, new_r = x;  // `new_r` starts as `x`
    
    while (new_r != 0)
    {
        u256 quotient = r / new_r;

        // Update t and new_t
        u256 temp_t = new_t;
        new_t = t - quotient * new_t;
        t = temp_t;
        
        // Update r and new_r
        u256 temp_r = new_r;
        new_r = r - quotient * new_r;
        r = temp_r;
    }
    
    // If gcd(x, m) != 1, modular inverse does not exist
    assert(r <= 1);

    // Ensure t is positive (mod m)
    if (t > m)
        t = t % m;
    else if (t < 0)  // Ensure no underflow
        t = t + m;
    
    return t;
}

__device__ __forceinline__
static u256 montgomery_mul(const u256& x, const u256& y, const u256& m)
{
    
    return m;
}

__device__ __forceinline__
static u256 add_modulo(const u256& lhs, const u256& rhs, const u256& m)
{
    assert(lhs < m);
    assert(rhs < m);
    
    u256 result = lhs + rhs;
    if (result < lhs)
        result -= m;
    else
        result += m;

    return result;
}

__device__ __forceinline__
static u256 sub_modulo(const u256& lhs, const u256& rhs, const u256& m)
{
    assert(lhs < m);
    assert(rhs < m);
    
    u256 result = lhs - rhs;
    if (lhs < rhs)
        result += m;
    else
        result -= m;

    return result;
}

__device__
static u256 euclidean_division(const u256& lhs, const u256& rhs, const u256& m)
{
    return lhs * inverse_mod(rhs, m) % m;
}

struct point
{
    u256 x, y;

    __device__ __forceinline__
    point operator+ (const point& rhs) const
    {
        u256 slope = euclidean_division((y - rhs.y), (x - rhs.x), P);
        u256 x_r = slope * slope - x - rhs.x;
        u256 y_r = y + slope * (x_r - x);

        return { x_r % P, -y_r % P };
    }

    __device__ __forceinline__
    point operator* (const u256& scalar)
    {

    }

};

}
