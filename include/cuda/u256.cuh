#pragma once

#include <cstdint>

namespace lumina::ecdsa
{

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

struct u256
{
    u64 data[4];

    u256()
    {}

    __host__ __device__
    u256(u64 u0, u64 u1 = 0, u64 u2 = 0, u64 u3 = 0)
    :   data{u0, u1, u2, u3}
    {}

    __host__ __device__
    u256(u64* data)
    :   data{data[0], data[1], data[2], data[3]}
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

            : "=l"(x.data[0]), "=l"(x.data[1]), "=l"(x.data[2]), "=l"(x.data[3])
            : "l"(data[0]), "l"(rhs.data[0]),
              "l"(data[1]), "l"(rhs.data[1]),
              "l"(data[2]), "l"(rhs.data[2]),
              "l"(data[3]), "l"(rhs.data[3])
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

            : "=l"(data[0]), "=l"(data[1]), "=l"(data[2]), "=l"(data[3])
            : "l"(rhs.data[0]),
              "l"(rhs.data[1]),
              "l"(rhs.data[2]),
              "l"(rhs.data[3])
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

            : "=l"(x.data[0]), "=l"(x.data[1]), "=l"(x.data[2]), "=l"(x.data[3])
            : "l"(data[0]), "l"(rhs.data[0]),
              "l"(data[1]), "l"(rhs.data[1]),
              "l"(data[2]), "l"(rhs.data[2]),
              "l"(data[3]), "l"(rhs.data[3])
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

            : "=l"(data[0]), "=l"(data[1]), "=l"(data[2]), "=l"(data[3])
            : "l"(rhs.data[0]),
              "l"(rhs.data[1]),
              "l"(rhs.data[2]),
              "l"(rhs.data[3])
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

            : "=l"(x.data[0]),  "=l"(x.data[1]),  "=l"(x.data[2]),  "=l"(x.data[3])
            : "l"(data[0]),     "l"(data[1]),     "l"(data[2]),     "l"(data[3]),
              "l"(rhs.data[0]), "l"(rhs.data[1]), "l"(rhs.data[2]), "l"(rhs.data[3])
        );

        return x;
    }

    __device__
    u256 operator/ (const u256& rhs) const
    {
        // Algorithm D
        assert(rhs != 0);

        // D1 Normlize

        u256 y_norm = rhs;
        u32 s = 0;
        while (not (y_norm[0] & 0x8000000000000000))
        {
            y_norm = y_norm << 1;
            s++;
        }

        u256 x_norm = (*this) << s;

        // D2 Initalize

        u256 q(0, 0, 0, 0), r = x_norm;
        // if (*this < y) return {q, *this}; // If x < y, quotient is 0, remainder is x

        // Main division loop
        for (int i = 256 - 1; i >= 0; --i) {
            // Shift remainder left and bring down next bit from x_norm
            r = r << 1;
            r[0] |= (x_norm[3] & (1ULL << 63)) >> 63; // Bring down the next bit
            x_norm = x_norm << 1;

            // Estimate quotient digit
            uint64_t qhat = (r[3] == y_norm[3])
                                ? ~0ULL
                                : (uint128_t(r[3]) << 64 | r[2]) / y_norm[3];

            // Multiply and subtract: r -= qhat * y_norm
            u256 prod = y_norm * qhat;
            if (prod > r) {
                --qhat;
                prod -= y_norm;
            }
            r -= prod;

            // Set the quotient bit
            q[i / 64] |= (qhat << (i % 64));
        }

        // Undo normalization on remainder
        r >>= shift;

        

        //

        return q;
    }

    __device__ __forceinline__
    u256 operator% (const u256& rhs) const
    {
        return (*this) - (*this) / rhs * rhs; 
    }

    // Unary

    __device__ __forceinline__
    u256 operator~ () const
    {
        return { ~data[0], ~data[1], ~data[2], ~data[3] };
    }

    __device__ __forceinline__
    u256 operator- () const
    {
        return ~(*this) + 1;
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

            : "=l"(x.data[0]), "=l"(x.data[1]), "=l"(x.data[2]), "=l"(x.data[3])
            : "l"(data[0]), "l"(data[1]), "l"(data[2]), "l"(data[3]), "r"(n), "r"(64 - n)
        );

        return x;
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

            : "=l"(x.data[0]), "=l"(x.data[1]), "=l"(x.data[2]), "=l"(x.data[3])
            : "l"(data[0]), "l"(data[1]), "l"(data[2]), "l"(data[3]), "r"(n), "r"(64 - n)
        );

        return x;
    }

    // Comparison
    
    __device__ __forceinline__
    bool operator> (const u256& rhs) const
    {
        #pragma unroll
        for(int i = 4; i >= 0; --i)
            if (data[i] > rhs[i])
                return true;

        return false;
    }

    __device__ __forceinline__
    bool operator== (const u256& rhs) const
    {
        #pragma unroll
        for(int i = 4; i >= 0; --i)
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
