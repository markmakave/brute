#include <iostream>
#include <string>

#include <cstdint>
#include <cstring>

#include <secp256k1.h>

#include <openssl/sha.h>
#include <openssl/ripemd.h>

#include <unistd.h>
#include <fcntl.h>

std::string public_key_generate(uint8_t privkey[32]);

void print(uint8_t* data, size_t size)
{
    for (int i = 0; i < size; ++i) printf("%02X", data[i]);
    printf("\n");
}


int main()
{

    uint8_t privkey[32] = {
        0x1C, 0xBA, 0xD4, 0x48, 0xF6, 0xAE, 0x48, 0x69,
        0x98, 0x4E, 0x4C, 0x26, 0x84, 0x13, 0xB2, 0xE3,
        0x03, 0xBE, 0xC4, 0xCC, 0x11, 0xCD, 0x83, 0x64,
        0x80, 0x91, 0x15, 0xFB, 0x2E, 0xB3, 0x6B, 0x35
    };
    

    #pragma omp parallel for
    for (int i = 0; i < 100; ++i)
        public_key_generate(privkey);

    return 0;
}


bool base58(char *b58, size_t *b58sz, const uint8_t *data, size_t binsz)
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


std::string public_key_generate(uint8_t privkey[32])
{

// Stage 1 /////////////////////////////////////////////////////////////////

    uint8_t stage1[65];

    //////////

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    if (!secp256k1_ec_seckey_verify(ctx, privkey))
    {
        std::cerr << "Invalid private key\n";
        return {};
    }
    
    secp256k1_pubkey pubkey;
    auto _ = secp256k1_ec_pubkey_create(ctx, &pubkey, privkey);
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

    std::string stage9;

    //////////

    char buffer[64] = {};
    size_t size = sizeof(buffer);
    base58(buffer, &size, stage8, sizeof(stage8));

    stage9.assign(buffer);

///////////////////////////////////////////////////////////////////////////

    return stage9;
}