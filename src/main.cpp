#include "credentials.hpp"
#include "blockchain.hpp"
#include "notifier.hpp"

int main() {

    btc::blockchain blockchain;

    int64_t iteration = 0;
    while (true) {
        iteration++;

        std::vector<btc::key> keys(400);
        std::vector<btc::address> addresses(keys.size());
        std::transform(keys.begin(), keys.end(), addresses.begin(), [](btc::key& k){ return btc::address(k); });

        // if (iteration == 5)
        //     addresses[30] = "bc1qn0h74msknqpsn8hgn4fght98mykcwkl5tse485";

        auto balances = blockchain.check_balance(addresses);

        for (int i = 0; i < balances.size(); ++i)
            if (balances[i] != 0) {
                std::stringstream message;
                message << "Fount non-zero balance\n"
                        "Key:     " << keys[i]      << "\n"
                        "Address: " << addresses[i] << "\n"
                        "Balance: " << balances[i]  << "\n";
                        
                btc::notifier::notify(827454744, message.str());

                exit(1);
            }

        if (iteration % 10000 == 0) {
            std::string message = std::to_string(iteration) + " iterations have passed";
            btc::notifier::notify(827454744, message);
        }
    }

    return 0;
}