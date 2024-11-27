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
        u64 _u64[4];
        u32 _u32[8];
        u16 _u16[16];
        u8  _u8[32];
    };

    __host__ __device__
    u256()
    {}

    __host__ __device__
    u256(u64 u0, u64 u1 = 0, u64 u2 = 0, u64 u3 = 0)
    :   _u64{u0, u1, u2, u3}
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

            : "=l"(x._u64[0]),
              "=l"(x._u64[1]),
              "=l"(x._u64[2]),
              "=l"(x._u64[3])

            : "l"(_u64[0]), "l"(rhs._u64[0]),
              "l"(_u64[1]), "l"(rhs._u64[1]),
              "l"(_u64[2]), "l"(rhs._u64[2]),
              "l"(_u64[3]), "l"(rhs._u64[3])
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

            : "+l"(_u64[0]),
              "+l"(_u64[1]),
              "+l"(_u64[2]),
              "+l"(_u64[3])

            : "l"(rhs._u64[0]),
              "l"(rhs._u64[1]),
              "l"(rhs._u64[2]),
              "l"(rhs._u64[3])
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

            : "=l"(x._u64[0]),
              "=l"(x._u64[1]),
              "=l"(x._u64[2]),
              "=l"(x._u64[3])

            : "l"(_u64[0]), "l"(rhs._u64[0]),
              "l"(_u64[1]), "l"(rhs._u64[1]),
              "l"(_u64[2]), "l"(rhs._u64[2]),
              "l"(_u64[3]), "l"(rhs._u64[3])
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

            : "+l"(_u64[0]),
              "+l"(_u64[1]),
              "+l"(_u64[2]),
              "+l"(_u64[3])

            : "l"(rhs._u64[0]),
              "l"(rhs._u64[1]),
              "l"(rhs._u64[2]),
              "l"(rhs._u64[3])
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

            : "=l"(x._u64[0]),
              "=l"(x._u64[1]),
              "=l"(x._u64[2]),
              "=l"(x._u64[3])

            : "l"(_u64[0]),     "l"(_u64[1]),     "l"(_u64[2]),     "l"(_u64[3]),
              "l"(rhs._u64[0]), "l"(rhs._u64[1]), "l"(rhs._u64[2]), "l"(rhs._u64[3])
        );

        return x;
    }

    __device__
    u256 operator/ (const u256& rhs) const
    {
        u256 div, _;
        div_mod(*this, rhs, div, _);
        return div;
    }

    __device__
    u256 operator% (const u256& rhs) const
    {
        u256 _, mod;
        div_mod(*this, rhs, _, mod);
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

        static constexpr u64 b = u64(1) << 32;

        assert(rhs != 0);
        if (lhs < rhs)
        {
            div = 0;
            mod = lhs;
            return;
        }

        u32 m, n;
        
        #pragma unroll 7
        for (m = 7; m > 0; --m)
            if (lhs._u32[m]) break;

        #pragma unroll 7
        for (n = 7; n > 0; --n)
            if (rhs._u32[n]) break;

        ++n; ++m;
        m = m - n;

        // D1

        u32 s;
        asm volatile (
            "bfind.shiftamt.u32 %0, %1;"
            : "=r"(s)
            : "r"(rhs._u32[n - 1])
        );
        assert(s < 32);

        union extension
        {
            u256 _u256;
            u64  _u64[5];
            u32  _u32[10];
        };

        extension u = {};
        u._u256 = lhs << s;
        u._u64[4] = lhs._u64[3] >> (64 - s);

        u256 v = rhs << s;
        assert(v._u32[n - 1] > (1 << 31));

        // D2

        extension q = {};

        #pragma unroll 8
        for (i32 j = m; j >= 0; --j)
        {
            // D3

            u64 wide = (static_cast<u64>(u._u32[j + n]) * b) | (static_cast<u64>(u._u32[j + n - 1]));
            u64 q_hat = wide / static_cast<u64>(v._u32[n - 1]);
            u64 r_hat = wide % static_cast<u64>(v._u32[n - 1]);

            if ((q_hat == b) or (q_hat * v._u32[n - 2] > (r_hat * b) + u._u32[j + n - 2]))
            {
                --q_hat;
                r_hat += v._u32[n - 1];
            }

            // D4

            extension q_hat_v = {};

            // q_hat_v = q_hat * v
            asm volatile (
                "mul.lo.u64     %0, %5, %9     ;"
                "mul.lo.u64     %1, %6, %9     ;"
                "mul.lo.u64     %2, %7, %9     ;"
                "mul.lo.u64     %3, %8, %9     ;"

                "mad.hi.cc.u64  %1, %5, %9, %1 ;"
                "madc.hi.cc.u64 %2, %6, %9, %2 ;"
                "madc.hi.cc.u64 %3, %7, %9, %3 ;"
                "madc.hi.u64    %4, %8, %9,  0 ;"
                
                : "=l"(q_hat_v._u64[0]),
                  "=l"(q_hat_v._u64[1]),
                  "=l"(q_hat_v._u64[2]),
                  "=l"(q_hat_v._u64[3]),
                  "=l"(q_hat_v._u64[4])

                : "l"(v._u64[0]),
                  "l"(v._u64[1]),
                  "l"(v._u64[2]),
                  "l"(v._u64[3]),
                  "l"(q_hat)
            );

            // u -= q_hat_v
            u32 borrow = false;
            asm volatile (
                "sub.cc.u64     %0, %0, %6 ;"
                "subc.cc.u64    %1, %1, %7 ;"
                "subc.cc.u64    %2, %2, %8 ;"
                "subc.cc.u64    %3, %3, %9 ;"
                "subc.cc.u64    %4, %4, %10;"
                "addc.u32       %5,  0,  0 ;"

                : "+l"(u._u64[0]),
                  "+l"(u._u64[1]),
                  "+l"(u._u64[2]),
                  "+l"(u._u64[3]),
                  "+l"(u._u64[4]),
                  "=r"(borrow)

                : "l"(q_hat_v._u64[0]),
                  "l"(q_hat_v._u64[1]),
                  "l"(q_hat_v._u64[2]),
                  "l"(q_hat_v._u64[3]),
                  "l"(q_hat_v._u64[4])
            );

            // D5

            assert(q_hat < (1ull << 32));
            q._u32[j] = q_hat;

            if (borrow)
            {
                printf("rare case\n");
                // D6
                q._u32[j] -= 1;

                // u += v
                asm volatile (
                    "add.cc.u64     %0, %0, %5 ;"
                    "addc.cc.u64    %1, %1, %6 ;"
                    "addc.cc.u64    %2, %2, %7 ;"
                    "addc.cc.u64    %3, %3, %8 ;"
                    "addc.u64       %4, %4,  0 ;"

                    : "+l"(u._u64[0]),
                      "+l"(u._u64[1]),
                      "+l"(u._u64[2]),
                      "+l"(u._u64[3]),
                      "+l"(u._u64[4])

                    : "l"(v._u64[0]),
                      "l"(v._u64[1]),
                      "l"(v._u64[2]),
                      "l"(v._u64[3])
                );
            }

            // D7
        }

        // D8
        
        div = q._u256;

        mod = 0;
        #pragma unroll 8
        for (u32 i = 0; i < n; ++i)
            mod._u32[i] = u._u32[i];
        mod >>= s;
    }

    // Unary

    __device__ __forceinline__
    u256 operator~ () const
    {
        return { ~_u64[0], ~_u64[1], ~_u64[2], ~_u64[3] };
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
            "{"
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
            "}"

            : "=l"(x._u64[0]),
              "=l"(x._u64[1]),
              "=l"(x._u64[2]),
              "=l"(x._u64[3])

            : "l"(_u64[0]),
              "l"(_u64[1]),
              "l"(_u64[2]),
              "l"(_u64[3]), 
              "r"(n), "r"(64 - n)
        );

        return x;
    }

    __device__
    u256& operator<<= (u32 n)
    {
        asm volatile (
            "{"
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
            "}"

            : "+l"(_u64[0]),
              "+l"(_u64[1]),
              "+l"(_u64[2]),
              "+l"(_u64[3])

            : "r"(n), "r"(64 - n)
        );

        return *this;
    }

    __device__
    u256 operator>> (u32 n) const
    {
        u256 x;

        asm volatile (
            "{"
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
            "}"

            : "=l"(x._u64[0]),
              "=l"(x._u64[1]),
              "=l"(x._u64[2]),
              "=l"(x._u64[3])

            : "l"(_u64[0]),
              "l"(_u64[1]),
              "l"(_u64[2]),
              "l"(_u64[3]),
              "r"(n), "r"(64 - n)
        );

        return x;
    }

    __device__
    u256& operator>>= (u32 n)
    {
        asm volatile (
            "{"
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
            "}"

            : "+l"(_u64[0]),
              "+l"(_u64[1]),
              "+l"(_u64[2]),
              "+l"(_u64[3])

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
            if (_u64[i] == rhs._u64[i])
                continue;
            return _u64[i] < rhs._u64[i];
        }

        return false;
    }

    __device__ __forceinline__
    bool operator== (const u256& rhs) const
    {
        #pragma unroll
        for(int i = 3; i >= 0; --i)
            if (_u64[i] != rhs._u64[i])
                return false;

        return true;
    }

    // utility

    __host__
    friend std::ostream& operator<< (std::ostream& os, const u256& x)
    {
        os << "0x";
        for (int i = 3; i >= 0; --i)
            os << std::setw(16) << std::setfill('0') << std::hex << x._u64[i];

        return os;
    }

};

}
