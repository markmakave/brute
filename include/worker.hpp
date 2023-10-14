#pragma once

#include <thread>
#include <atomic>
#include <algorithm>
#include <iomanip>

#include "credentials.hpp"
#include "database.hpp"
#include "notifier.hpp"

namespace btc {

class worker {
public:

    worker(const database& db, int batch) : _db(&db), _batch(batch) {}

    worker(worker&& w) : _db(w._db), _batch(w._batch) {
        w.stop();
    }

    ~worker() {
        stop();
    }

    void run() {
        _stop = false;
        _thread = std::thread(&worker::loop, this);
    }

    void stop() {
        _stop = true;
        if (_thread.joinable())
            _thread.join();
    }

    int64_t iterations() const {
        return _iterations;
    }

    double time() const {
        double avg = 0;
        for (size_t i = 0; i < _time.size(); ++i)
            avg += _time[i];
        avg /= _time.size();
        avg /= 1000;
        return avg;
    }

private:

    void loop() {
        while (!_stop) {
            _iterations++;

            auto now = std::chrono::system_clock::now();

            std::vector<btc::key> keys(_batch);
            std::vector<btc::address> addresses(_batch);
            std::transform(keys.begin(), keys.end(), addresses.begin(), [](btc::key& k){ return btc::address(k); });

            // if (iteration == 5)
            //     addresses[30] = "bc1qn0h74msknqpsn8hgn4fght98mykcwkl5tse485";

            auto balances = _db->check_balance(addresses);

            for (size_t i = 0; i < balances.size(); ++i)
                if (balances[i] != 0) {
                    std::stringstream message;
                    message << "Found non-zero balance\n"
                            "Key:     " << keys[i]      << "\n"
                            "Address: " << addresses[i] << "\n"
                            "Balance: " << balances[i]  << "\n";
                            
                    btc::notifier::notify(827454744, message.str());
                }

            // Collect metrics
            int64_t time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - now).count();
            _time[_time_pos++] = time;
            _time_pos %= _time.size();
        }
    }

protected:
    const database* _db;
    int _batch;

    std::atomic<bool> _stop;
    std::thread _thread;

    int64_t _iterations = 0;
    size_t _time_pos = 0;
    std::array<int64_t, 100> _time;
}; 

}
