#pragma once

#include <map>
#include <fstream>
#include <cassert>

#include "credentials.hpp"

namespace btc {

class database {
public:

    database(const char* filename) {
        std::ifstream file(filename);
        assert(file.is_open());

        std::cout << "Indexing " << filename << std::endl;

        std::string line;
        std::getline(file, line);
        while (std::getline(file, line)) {
            size_t delim = line.find('\t');
            if (delim == std::string::npos)
                continue;

            address addr = line.substr(0, delim).c_str();
            int64_t value = std::stoll(line.substr(delim + 1));

            _cache[addr] = value;
        }

        std::cout << "Done indexing " << filename << std::endl;
    }

    int64_t operator [] (const address& addr) const {
        auto it = _cache.find(addr);
        if (it == _cache.end())
            return 0;

        return it->second;
    }

    std::vector<int64_t> check_balance(const std::vector<address>& addresses) const {
        std::vector<int64_t> balances(addresses.size());
        for (int i = 0; i < addresses.size(); ++i)
            balances[i] = operator[](addresses[i]);

        return balances;
    }

protected:

    std::map<address, int64_t> _cache;
};

}
