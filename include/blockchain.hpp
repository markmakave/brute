#pragma once

#include <thread>
#include <mutex>

#include "credentials.hpp"

#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>

#include <nlohmann/json.hpp>

#include "notifier.hpp"

namespace btc {

class blockchain {
public:

    blockchain() {
        curlpp::initialize(); 
    }

    ~blockchain() {
        curlpp::terminate();
        if (_timer.joinable())
            _timer.join();
    }

    std::vector<int64_t> check_balance(const std::vector<address>& addresses)
    {
        std::string url = "https://blockchain.info/balance?active=";
        for (const auto& address : addresses)
            url += "|" + static_cast<std::string>(address);

        nlohmann::json json;

        bool ok = false;
        do {
            curlpp::Easy request;

            request.setOpt(new curlpp::options::Url(url));
            std::stringstream response;
            request.setOpt(new curlpp::options::WriteStream(&response));

            if (_timer.joinable())
                _timer.join();

            try {
                request.perform();
            } catch(...) {
                notifier::notify(827454744, "CURL Error:\n" + response.str());
                continue;
            }
            
            _timer = std::thread([&](){
                std::this_thread::sleep_for(std::chrono::seconds(1));
            });

            try {
                json = nlohmann::json::parse(response);
                ok = true;
            } catch (...) {
                notifier::notify(827454744, "JSON Error:\n" + response.str());
                continue;
            }
            
        } while (!ok);

        std::vector<int64_t> balances;
        for (const auto& address : addresses) {
            auto& entry = json[static_cast<std::string>(address)];
            balances.push_back(entry["final_balance"].get<int64_t>());
        }

        return balances;
    }

protected:

    std::thread _timer;
};

}
