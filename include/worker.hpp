#pragma once

#include <thread>
#include <atomic>
#include <algorithm>
#include <iomanip>
#include <mutex>
#include <set>

#include "credentials.hpp"
#include "database.hpp"
#include "notifier.hpp"

std::set<btc::key> checked;
std::mutex mutex;

namespace btc {

class worker {
public:

    worker(const database& db) : _db(&db) {}

    worker(worker&& w) : _db(w._db) {
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

    int total() {
        return _iterations;
    }

    int throughput() {
        return _throughput;
    }

private:

    void loop() {
        _time = std::chrono::high_resolution_clock::now();

        while (not _stop) {
            _iterations++;

            auto now = std::chrono::system_clock::now();

            btc::key key;
            btc::address address(key);

            // if (_iterations == 500000)
            //     address = "bc1qn0h74msknqpsn8hgn4fght98mykcwkl5tse485";

            auto balance = (*_db)[address];

            if (balance != 0) {
                std::stringstream message;
                message << "Found non-zero balance\n"
                        "Key:     " << key     << "\n"
                        "Address: " << address << "\n"
                        "Balance: " << balance  << "\n";
                        
                btc::notifier::notify(827454744, message.str());
            }

            if (_iterations % 1000 == 0) {
                int64_t delay = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - _time).count();
                _throughput = 1000 / (delay / 1000000.0);
                _time = std::chrono::high_resolution_clock::now();
            }
        }
    }

protected:
    const database* _db;

    std::atomic<bool> _stop;
    std::thread _thread;

    int64_t _iterations = 0, _throughput = 0;
    std::chrono::high_resolution_clock::time_point _time;
}; 

}
