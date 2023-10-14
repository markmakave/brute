#include "database.hpp"
#include "worker.hpp"

#include <chrono>

#define BATCH 1000

int main(int argc, char** argv) {
    int nworkers = 4;
    if (argc > 1)
        nworkers = std::atoi(argv[1]);

    btc::database db("../resource/blockchair_bitcoin_addresses_latest.tsv");

    std::vector<btc::worker> workers;
    for (int i = 0; i < nworkers; ++i)
        workers.emplace_back(db, BATCH);

    for (auto& w : workers)
        w.run();

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        int64_t total_checked = 0;
        double avg_time = 0;
        
        for (auto& w : workers) {
            total_checked += w.iterations();
            avg_time += w.time();
        }
        total_checked *= BATCH;
        avg_time /= workers.size();

        std::cout << "\033[2JTotal checked: " << total_checked << std::endl;
        std::cout << "Throughput: " << BATCH * workers.size() / avg_time << " tx/s" << std::endl;
    }

    for (auto& w : workers)
        w.stop();

    return 0;
}