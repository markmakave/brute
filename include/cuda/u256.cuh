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
    union {
        u64 u64_data[4];
        u32 u32_data[8];
        u16 u16_data[16];
        u8  u8_data[32];
    };

    __host__ __device__
    u256()
    {}

    __host__ __device__
    u256(u64 u0, u64 u1 = 0, u64 u2 = 0, u64 u3 = 0)
    :   u64_data{u0, u1, u2, u3}
    {}

    // Binary

    __device__ __forceinline__
    u256 operator+ (const u256& rhs) const
    {
        u256 x;

        asm volatile (
            "add.cc.u64  %0, %4,  %5 ;"
            "addc.cc.u64 %1, %6,  %7 ;"
            "addc.cc.u64 %2, %8,  %9 ;"
            "addc.u64    %3, %10, %11;"

            : "=l"(x.u64_data[0]), "=l"(x.u64_data[1]), "=l"(x.u64_data[2]), "=l"(x.u64_data[3])
            : "l"(u64_data[0]), "l"(rhs.u64_data[0]),
              "l"(u64_data[1]), "l"(rhs.u64_data[1]),
              "l"(u64_data[2]), "l"(rhs.u64_data[2]),
              "l"(u64_data[3]), "l"(rhs.u64_data[3])
        );

        return x;
    }

    __device__ __forceinline__
    u256& operator+= (const u256& rhs)
    {
        asm volatile (
            "add.cc.u64  %0, %0, %4 ;"
            "addc.cc.u64 %1, %1, %5 ;"
            "addc.cc.u64 %2, %2, %6 ;"
            "addc.u64    %3, %3, %7 ;"

            : "=l"(u64_data[0]), "=l"(u64_data[1]), "=l"(u64_data[2]), "=l"(u64_data[3])
            : "l"(rhs.u64_data[0]),
              "l"(rhs.u64_data[1]),
              "l"(rhs.u64_data[2]),
              "l"(rhs.u64_data[3])
        );

        return *this;
    }

    __device__ __forceinline__
    u256 operator- (const u256& rhs) const
    {
        u256 x;

        asm volatile (
            "sub.cc.u64  %0, %4,  %5 ;"
            "subc.cc.u64 %1, %6,  %7 ;"
            "subc.cc.u64 %2, %8,  %9 ;"
            "subc.u64    %3, %10, %11;"

            : "=l"(x.u64_data[0]), "=l"(x.u64_data[1]), "=l"(x.u64_data[2]), "=l"(x.u64_data[3])
            : "l"(u64_data[0]), "l"(rhs.u64_data[0]),
              "l"(u64_data[1]), "l"(rhs.u64_data[1]),
              "l"(u64_data[2]), "l"(rhs.u64_data[2]),
              "l"(u64_data[3]), "l"(rhs.u64_data[3])
        );

        return x;
    }

    __device__ __forceinline__
    u256& operator-= (const u256& rhs)
    {
        asm volatile (
            "sub.cc.u64  %0, %0, %4 ;"
            "subc.cc.u64 %1, %1, %5 ;"
            "subc.cc.u64 %2, %2, %6 ;"
            "subc.u64    %3, %3, %7 ;"

            : "=l"(u64_data[0]),
              "=l"(u64_data[1]),
              "=l"(u64_data[2]),
              "=l"(u64_data[3])

            : "l"(rhs.u64_data[0]),
              "l"(rhs.u64_data[1]),
              "l"(rhs.u64_data[2]),
              "l"(rhs.u64_data[3])
        );

        return *this;
    }

    __device__ __forceinline__
    u256 operator* (const u256& rhs) const
    {
        u256 x;

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

            : "=l"(x.u64_data[0]),
              "=l"(x.u64_data[1]),
              "=l"(x.u64_data[2]),
              "=l"(x.u64_data[3])

            : "l"(u64_data[0]),     "l"(u64_data[1]),     "l"(u64_data[2]),     "l"(u64_data[3]),
              "l"(rhs.u64_data[0]), "l"(rhs.u64_data[1]), "l"(rhs.u64_data[2]), "l"(rhs.u64_data[3])
        );

        return x;
    }

    __device__
    u256 operator/ (const u256& rhs) const
    {
        u256 div, mod;
        // knuth::D(*this, rhs, div, mod);
        div_mod(*this, rhs, div, mod);
        return div;
    }

    __device__
    u256 operator% (const u256& rhs) const
    {
        u256 div, mod;
        // knuth::D(*this, rhs, div, mod);
        div_mod(*this, rhs, div, mod);
        return mod;
    }

    __device__
    static void div_mod(
        const u256& lhs,
        const u256& rhs,
              u256& div,
              u256& mod
    ) {
        // Donald Knuth's Algorithm D
        assert(rhs != 0);
        if (lhs < rhs)
        {
            div = 0;
            mod = lhs;
            return;
        }

        u32 m, n;
        
        for (m = 7; m > 0; --m)
            if (lhs.u32_data[m]) break;

        for (n = 7; n > 0; --n)
            if (rhs.u32_data[n]) break;

        ++n; ++m;
        m = m - n;

        // D1 Normalize

        u32 s;
        asm volatile (
            "bfind.shiftamt.u32 %0, %1;"
            : "=r"(s)
            :"r"(rhs.u32_data[n - 1])
        );
        assert(s < 32);
        printf("s: %u\n", s);

        u32 extension = lhs.u32_data[m - n] >> s;
        
    }

    // Unary

    __device__ __forceinline__
    u256 operator~ () const
    {
        return { ~u64_data[0], ~u64_data[1], ~u64_data[2], ~u64_data[3] };
    }

    __device__ __forceinline__
    u256 operator- () const
    {
        return ~(*this) + 1;
    }

    __device__
    u256& operator++ ()
    {
        return *this += 1;
    }

    // Bitshift

    __device__
    u256 operator<< (u32 n) const
    {
        u256 x;

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

            : "=l"(x.u64_data[0]),
              "=l"(x.u64_data[1]),
              "=l"(x.u64_data[2]),
              "=l"(x.u64_data[3])

            : "l"(u64_data[0]),
              "l"(u64_data[1]),
              "l"(u64_data[2]),
              "l"(u64_data[3]), 
              "r"(n), "r"(64 - n)
        );

        return x;
    }

    __device__
    u256& operator<<= (u32 n)
    {
        asm volatile (
            ".reg.b64   %r          ;"

            "shl.b64    %3, %3, %8  ;"

            "shr.b64    %r, %2, %5  ;"
            " or.b64    %3, %3, %r  ;"
            "shl.b64    %2, %2, %4  ;"

            "shr.b64    %r, %1, %5  ;"
            " or.b64    %2, %2, %r  ;"
            "shl.b64    %1, %1, %4  ;"

            "shr.b64    %r, %0, %5  ;"
            " or.b64    %1, %1, %r  ;"
            "shl.b64    %0, %0, %4  ;"

            : "+l"(u64_data[0]),
              "+l"(u64_data[1]),
              "+l"(u64_data[2]),
              "+l"(u64_data[3])

            : "r"(n), "r"(64 - n)
        );

        return *this;
    }

    __device__
    u256 operator>> (u32 n) const
    {
        u256 x;

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

            : "=l"(x.u64_data[0]),
              "=l"(x.u64_data[1]),
              "=l"(x.u64_data[2]),
              "=l"(x.u64_data[3])

            : "l"(u64_data[0]),
              "l"(u64_data[1]),
              "l"(u64_data[2]),
              "l"(u64_data[3]),
              "r"(n), "r"(64 - n)
        );

        return x;
    }

    __device__
    u256& operator>>= (u32 n)
    {
        asm volatile (
            ".reg.b64   %r          ;"

            "shr.b64    %0, %0, %4  ;"

            "shl.b64    %r, %1, %5  ;"
            " or.b64    %0, %0, %r  ;"
            "shr.b64    %1, %1, %4  ;"

            "shl.b64    %r, %2, %5  ;"
            " or.b64    %1, %1, %r  ;"
            "shr.b64    %2, %2, %4  ;"

            "shl.b64    %r, %3, %5  ;"
            " or.b64    %2, %2, %r  ;"
            "shr.b64    %3, %3, %4  ;"

            : "+l"(u64_data[0]),
              "+l"(u64_data[1]),
              "+l"(u64_data[2]),
              "+l"(u64_data[3])

            : "r"(n), "r"(64 - n)
        );

        return *this;
    }

    // Comparison
    
    __device__ __forceinline__
    bool operator< (const u256& rhs) const
    {
        #pragma unroll
        for(int i = 3; i >= 0; --i)
        {
            if (u64_data[i] == rhs.u64_data[i])
                continue;
            return u64_data[i] < rhs.u64_data[i];
        }

        return false;
    }

    __device__ __forceinline__
    bool operator== (const u256& rhs) const
    {
        #pragma unroll
        for(int i = 3; i >= 0; --i)
            if (u64_data[i] != rhs.u64_data[i])
                return false;

        return true;
    }

    // utility

    __host__
    friend std::ostream& operator<< (std::ostream& os, const u256& x)
    {
        os << "0x";
        for (int i = 3; i >= 0; --i)
            os << std::setw(16) << std::setfill('0') << std::hex << x.u64_data[i];

        return os;
    }

};

}
