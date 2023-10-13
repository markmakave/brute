#pragma once

#include <iostream>
#include <iomanip>
#include <random>
#include <cstdint>
#include <cstring>
#include <secp256k1.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>

namespace btc {

void hash(const uint8_t* key, char* address);

class key {
    friend class address;
public:

    key() {
        uint64_t* field = (uint64_t*)_data;

        static std::random_device rd;
        static std::mt19937_64 gen(rd());
        static std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);

        for (int i = 0; i < 4; ++i)
            field[i] = dis(gen);
    }

    friend std::ostream& operator << (std::ostream& stream, const key& k) {
        for (int i = 0; i < sizeof(k._data); ++i)
            stream << std::hex << std::setfill('0') << std::setw(2) << int(k._data[i]);
        return stream << std::dec;
    }

    friend bool operator < (const key& k1, const key& k2) {
        uint64_t* f1 = (uint64_t*)k1._data;
        uint64_t* f2 = (uint64_t*)k2._data;

        for (int i = 0; i < 4; ++i)
            if (f1[i] < f2[i])
                return true;
        return false;
    }

protected:

    uint8_t _data[32];
};

class address {
public:

    address() : _data{} {}

    address(const key& k) {
        hash(k._data, _data);
    }

    operator std::string() const {
        return _data;
    }

    friend std::ostream& operator << (std::ostream& stream, const address& a) {
        return stream << a._data;
    }

    void operator = (const char* s) {
        int len = std::strlen(s);
        for (int i = 0; i < len; ++i)
            _data[i] = s[i];
        _data[len] = 0;
    }

protected:

    char _data[64];
};

static bool base58(char *b58, size_t *b58sz, const uint8_t *data, size_t binsz)
{
    static const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    const uint8_t *bin = data;
    int carry;
    ssize_t i, j, high, zcount = 0;
    uint8_t buf[12 * 1024] = {0};
    size_t size;

    // Рассчитать количество начальных нулей данных для кодирования 
    while (zcount < (ssize_t)binsz && !bin[zcount])
        ++zcount;

    // Рассчитать размер массива, необходимого для хранения преобразованных данных 138 / 100-> log (256) / log (58)
    size = (binsz - zcount) * 138 / 100 + 1;
    memset(buf, 0, size);
    
    // Обходим данные для преобразования
    for (i = zcount, high = size - 1; i < (ssize_t)binsz; ++i, high = j)
    {
        // Сохраняем данные последовательно от начала до конца
        for (carry = bin[i], j = size - 1; (j > high) || carry; --j)
        {
            carry += 256 * buf[j];
            buf[j] = carry % 58;
            carry /= 58;
        }
    }

    for (j = 0; j < (ssize_t)size && !buf[j]; ++j);

    if (*b58sz <= zcount + size - j)
    {
        *b58sz = zcount + size - j + 1;
        return false;
    }

    if (zcount)
        memset(b58, '1', zcount);
    for (i = zcount; j < (ssize_t)size; ++i, ++j)
        b58[i] = b58digits_ordered[buf[j]];
    b58[i] = '\0';
    *b58sz = i + 1;

    return true;
}

void hash(const uint8_t* key, char* address) {

// Stage 1 /////////////////////////////////////////////////////////////////

    uint8_t stage1[65];

    //////////

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    if (!secp256k1_ec_seckey_verify(ctx, key))
    {
        std::cerr << "Invalid private key\n";
        return; 
    }
    
    secp256k1_pubkey pubkey;
    auto _ = secp256k1_ec_pubkey_create(ctx, &pubkey, key);
    size_t outlen = 65;
    secp256k1_ec_pubkey_serialize(ctx, stage1, &outlen, &pubkey, SECP256K1_EC_UNCOMPRESSED);

    secp256k1_context_destroy(ctx);

// Stage 2 /////////////////////////////////////////////////////////////////

    uint8_t stage2[32];

    //////////

    SHA256(stage1, sizeof(stage1), stage2);

// Stage 3 /////////////////////////////////////////////////////////////////

    uint8_t stage3[20];

    //////////

    RIPEMD160(stage2, sizeof(stage2), stage3);

// Stage 4 /////////////////////////////////////////////////////////////////

    uint8_t stage4[21];

    //////////

    stage4[0] = 0x00;
    for (size_t i = 0; i < sizeof(stage3); ++i) stage4[i + 1] = stage3[i];

// Stage 5 /////////////////////////////////////////////////////////////////

    uint8_t stage5[32];

    //////////

    SHA256(stage4, sizeof(stage4), stage5);

// Stage 6 /////////////////////////////////////////////////////////////////

    uint8_t stage6[32];

    //////////

    SHA256(stage5, sizeof(stage5), stage6);

// Stage 7 /////////////////////////////////////////////////////////////////

    uint8_t stage7[4];

    //////////

    for (size_t i = 0; i < sizeof(stage7); ++i) stage7[i] = stage6[i];

// Stage 8 /////////////////////////////////////////////////////////////////

    uint8_t stage8[25];

    //////////

    for (size_t i = 0; i < sizeof(stage4); ++i) stage8[i] = stage4[i];
    for (size_t i = 0; i < sizeof(stage7); ++i) stage8[i + sizeof(stage4)] = stage7[i];

// Stage 9 /////////////////////////////////////////////////////////////////

    size_t size = 64;
    base58(address, &size, stage8, sizeof(stage8));

}

}
