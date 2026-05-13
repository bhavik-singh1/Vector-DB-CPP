#pragma once
#include <unordered_map>
#include <mutex>
#include <chrono>
#include "Core/Types.h"
#include "Algorithms/BruteForce.h"
#include "Algorithms/KDTree.h"
#include "Algorithms/HNSW.h"
#include "Core/StorageManager.h"

class VectorDB {
    std::unordered_map<int, VectorItem> store;
    BruteForce bf;
    KDTree     kdt;
    HNSW       hnsw;
    StorageManager storage;
    std::mutex mu;
    int nextId = 1;

public:
    const int dims;
    explicit VectorDB(int d);

    int insert(const std::string& meta, const std::string& cat,
               const std::vector<float>& emb, DistFn dist);
    bool remove(int id);

    struct Hit { int id; std::string meta, cat; std::vector<float> emb; float dist; };
    struct SearchOut { std::vector<Hit> hits; long long us; std::string algo, metric; };

    SearchOut search(const std::vector<float>& q, int k,
                     const std::string& metric, const std::string& algo);

    struct BenchOut { long long bfUs, kdUs, hnswUs; int n; };
    BenchOut benchmark(const std::vector<float>& q, int k, const std::string& metric);

    std::vector<VectorItem> all();
    HNSW::GraphInfo hnswInfo();
    size_t size();
};
