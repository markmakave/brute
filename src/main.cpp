#include "database.hpp"
#include "worker.hpp"
#include "notifier.hpp"

#include <chrono>

#define BATCH 1000

int main(int argc, char** argv) {
    int nworkers = 1;
    if (argc > 1)
        nworkers = std::atoi(argv[1]);

    const char* db_path = "../resource/blockchair_bitcoin_addresses_latest.tsv";
    if (argc > 2)
        db_path = argv[2];

    btc::database db(db_path);

    std::vector<btc::worker> workers;
    for (int i = 0; i < nworkers; ++i)
        workers.emplace_back(db);

    btc::notifier::notify(827454744, "Parsing started on " + std::to_string(nworkers) + " workers");

    for (auto& w : workers)
        w.run();

    std::chrono::system_clock::time_point time = std::chrono::system_clock::now();
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        int64_t total = 0,
                throughput = 0;

        for (auto& w : workers) {
            total += w.total();
            throughput += w.throughput();
        }

        std::cout << "\033[2JTotal checked: " << total << std::endl;
        std::cout << "Throughput: " << throughput << " tx/s" << std::endl;

        int delay = std::chrono::duration_cast<std::chrono::hours>(std::chrono::high_resolution_clock::now() - time).count();
        if (delay >= 1) {
            std::stringstream message;
            message << "Total checked: " << total << '\n'
                    << "Throughput: " << throughput << " tx/s";
                    
            btc::notifier::notify(827454744, message.str());

            time = std::chrono::system_clock::now();
        }
    }

    for (auto& w : workers)
        w.stop();

    return 0;
}