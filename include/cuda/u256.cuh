#pragma once

#include <iostream>
#include <cstdint>
#include <cassert>

#include <cuda_runtime.h>

namespace lumina::ecdsa
{

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

struct u256
{
    u64 data[4];

    u256()
    {}

    __host__ __device__
    u256(u64 u0, u64 u1 = 0, u64 u2 = 0, u64 u3 = 0)
    :   data{u0, u1, u2, u3}
    {}

    // Binary

    __host__ __device__ __forceinline__
    u256 operator+ (const u256& rhs) const
    {
        u256 x;

        #ifdef __CUDA_ARCH__
        asm volatile (
            "add.cc.u64  %0, %4,  %5 ;"
            "addc.cc.u64 %1, %6,  %7 ;"
            "addc.cc.u64 %2, %8,  %9 ;"
            "addc.u64    %3, %10, %11;"

            : "=l"(x.data[0]), "=l"(x.data[1]), "=l"(x.data[2]), "=l"(x.data[3])
            : "l"(data[0]), "l"(rhs.data[0]),
              "l"(data[1]), "l"(rhs.data[1]),
              "l"(data[2]), "l"(rhs.data[2]),
              "l"(data[3]), "l"(rhs.data[3])
        );
        #else
        bool carry = false;
        for (int i = 0; i < 4; ++i)
        {
            x.data[i] = data[i] + rhs.data[i] + carry;
            carry = (data[i] + rhs.data[i]) < data[0];
        }
        #endif

        return x;
    }

    __host__ __device__ __forceinline__
    u256& operator+= (const u256& rhs)
    {
        #ifdef __CUDA_ARCH__
        asm volatile (
            "add.cc.u64  %0, %0, %4 ;"
            "addc.cc.u64 %1, %1, %5 ;"
            "addc.cc.u64 %2, %2, %6 ;"
            "addc.u64    %3, %3, %7 ;"

            : "=l"(data[0]), "=l"(data[1]), "=l"(data[2]), "=l"(data[3])
            : "l"(rhs.data[0]),
              "l"(rhs.data[1]),
              "l"(rhs.data[2]),
              "l"(rhs.data[3])
        );
        #else
        *this = *this + rhs;
        #endif

        return *this;
    }

    __host__ __device__ __forceinline__
    u256 operator- (const u256& rhs) const
    {
        u256 x;

        #ifdef __CUDA_ARCH__
        asm volatile (
            "sub.cc.u64  %0, %4,  %5 ;"
            "subc.cc.u64 %1, %6,  %7 ;"
            "subc.cc.u64 %2, %8,  %9 ;"
            "subc.u64    %3, %10, %11;"

            : "=l"(x.data[0]), "=l"(x.data[1]), "=l"(x.data[2]), "=l"(x.data[3])
            : "l"(data[0]), "l"(rhs.data[0]),
              "l"(data[1]), "l"(rhs.data[1]),
              "l"(data[2]), "l"(rhs.data[2]),
              "l"(data[3]), "l"(rhs.data[3])
        );
        #else
        bool borrow = false;
        for (int i = 0; i < 4; ++i)
        {
            x.data[i] = data[i] - rhs.data[i] - borrow;
            borrow = (data[i] - rhs.data[i]) > data[0];
        }
        #endif

        return x;
    }

    __host__ __device__ __forceinline__
    u256& operator-= (const u256& rhs)
    {
        #ifdef __CUDA_ARCH__
        asm volatile (
            "sub.cc.u64  %0, %0, %4 ;"
            "subc.cc.u64 %1, %1, %5 ;"
            "subc.cc.u64 %2, %2, %6 ;"
            "subc.u64    %3, %3, %7 ;"

            : "=l"(data[0]), "=l"(data[1]), "=l"(data[2]), "=l"(data[3])
            : "l"(rhs.data[0]),
              "l"(rhs.data[1]),
              "l"(rhs.data[2]),
              "l"(rhs.data[3])
        );
        #else
        *this = *this - rhs;
        #endif

        return *this;
    }

    __host__ __device__ __forceinline__
    u256 operator* (const u256& rhs) const
    {
        u256 x;

        #ifdef __CUDA_ARCH__
        asm volatile (
            "mul.lo.u64     %0, %4, %8      ;"
            "mul.lo.u64     %1, %5, %8      ;"
            "mul.lo.u64     %2, %6, %8      ;"
            "mul.lo.u64     %3, %7, %8      ;"
            "mad.hi.cc.u64  %1, %4, %8,  %1 ;"
            "madc.hi.cc.u64 %2, %5, %8,  %2 ;"
            "madc.hi.u64    %3, %6, %8,  %3 ;"

            "mad.lo.cc.u64  %1, %4, %9,  %1 ;"
            "madc.lo.cc.u64 %2, %5, %9,  %2 ;"
            "madc.lo.u64    %3, %6, %9,  %3 ;"
            "mad.hi.cc.u64  %2, %4, %9,  %2 ;"
            "madc.hi.u64    %3, %4, %9,  %3 ;"

            "mad.lo.cc.u64  %2, %4, %10, %2 ;"
            "madc.lo.u64    %3, %5, %10, %3 ;"
            "mad.hi.u64     %3, %4, %10, %3 ;"

            "mad.lo.u64     %3, %4, %11, %3 ;"

            : "=l"(x.data[0]),  "=l"(x.data[1]),  "=l"(x.data[2]),  "=l"(x.data[3])
            : "l"(data[0]),     "l"(data[1]),     "l"(data[2]),     "l"(data[3]),
              "l"(rhs.data[0]), "l"(rhs.data[1]), "l"(rhs.data[2]), "l"(rhs.data[3])
        );
        #else
        // #error "TODO"
        #endif

        return x;
    }

    __host__ __device__
    u256 operator/ (const u256& rhs) const
    {
        #ifdef __CUDA_ARCH__

        asm volatile (
            ".reg .pred p   ;"
            ""
            "bfind."

            :
            :
        )

        #else
        // Donald Knuth's Algorithm D
        assert(rhs != 0);
        if (*this < rhs)
            return 0;

        u32 m, n;
        
        // PTX: use bfind

        for (m = 3; m > 0; --m)
            if (data[m]) break;

        for (n = 3; n > 0; --n)
            if (rhs[n]) break;

        ++n; ++m;
        m = m - n;

        // D1 Normalize


        std::cout << "m: " << m << ", n: " << n << '\n';

        #endif
    }

    __host__ __device__
    static void div_mod(
        const u256& x,
        const u256& y,
              u256& div,
              u256& mod
    ) {
        // Algorithm D
    }

    __device__ __forceinline__
    u256 operator% (const u256& rhs) const
    {
        return (*this) - (*this) / rhs * rhs; 
    }

    // Unary

    __host__ __device__ __forceinline__
    u256 operator~ () const
    {
        return { ~data[0], ~data[1], ~data[2], ~data[3] };
    }

    __host__ __device__ __forceinline__
    u256 operator- () const
    {
        return ~(*this) + 1;
    }

    __host__ __device__
    u256& operator++ ()
    {
        
    }

    // Bitshift

    __host__ __device__
    u256 operator<< (u32 n) const
    {
        u256 x;

        #ifdef __CUDA_ARCH__
        asm volatile (
            ".reg.b64   %r          ;"

            "shl.b64    %3, %7, %8  ;"

            "shr.b64    %r, %6, %9  ;"
            " or.b64    %3, %3, %r  ;"
            "shl.b64    %2, %6, %8  ;"

            "shr.b64    %r, %5, %9  ;"
            " or.b64    %2, %2, %r  ;"
            "shl.b64    %1, %5, %8  ;"

            "shr.b64    %r, %4, %9  ;"
            " or.b64    %1, %1, %r  ;"
            "shl.b64    %0, %4, %8  ;"

            : "=l"(x.data[0]), "=l"(x.data[1]), "=l"(x.data[2]), "=l"(x.data[3])
            : "l"(data[0]), "l"(data[1]), "l"(data[2]), "l"(data[3]), "r"(n), "r"(64 - n)
        );
        #else
        for (i32 i = 3; i >= 0; i--)
        {
            x[i] = data[i] << n;
            if (i)
                x[i] |= data[i - 1] >> (64 - n);
        }
        #endif

        return x;
    }

    __host__ __device__
    u256 operator>> (u32 n) const
    {
        u256 x;

        #ifdef __CUDA_ARCH__
        asm volatile (
            ".reg.b64   %r          ;"

            "shr.b64    %0, %4, %8  ;"

            "shl.b64    %r, %5, %9  ;"
            " or.b64    %0, %0, %r  ;"
            "shr.b64    %1, %5, %8  ;"

            "shl.b64    %r, %6, %9  ;"
            " or.b64    %1, %1, %r  ;"
            "shr.b64    %2, %6, %8  ;"

            "shl.b64    %r, %7, %9  ;"
            " or.b64    %2, %2, %r  ;"
            "shr.b64    %3, %7, %8  ;"

            : "=l"(x.data[0]), "=l"(x.data[1]), "=l"(x.data[2]), "=l"(x.data[3])
            : "l"(data[0]), "l"(data[1]), "l"(data[2]), "l"(data[3]), "r"(n), "r"(64 - n)
        );
        #else
        for (i32 i = 0; i < 4; i++)
        {
            x[i] = data[i] >> n;
            if (i < 3)
                x[i] |= data[i + 1] << (64 - n);
        }
        #endif

        return x;
    }

    // Comparison
    
    __host__ __device__ __forceinline__
    bool operator< (const u256& rhs) const
    {
        #pragma unroll
        for(int i = 3; i >= 0; --i)
        {
            if (data[i] == rhs[i])
                continue;
            return data[i] < rhs[i];
        }

        return false;
    }

    __host__ __device__ __forceinline__
    bool operator== (const u256& rhs) const
    {
        #pragma unroll
        for(int i = 3; i >= 0; --i)
            if (data[i] != rhs[i])
                return false;

        return true;
    }

    // utility

    __host__ __device__ __forceinline__
    u64& operator[] (size_t index)
    {
        return data[index];
    }

    __host__ __device__ __forceinline__
    const u64& operator[] (size_t index) const
    {
        return data[index];
    }

    __host__
    friend std::ostream& operator<< (std::ostream& os, const u256& x)
    {
        os << "0x";
        for (int i = 3; i >= 0; --i)
            os << std::setw(16) << std::setfill('0') << std::hex << x.data[i];
        return os;
    }

};

}
