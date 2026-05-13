#include "Core/VectorDB.h"
#include <iostream>

VectorDB::VectorDB(int d) : kdt(d), hnsw(16, 200), dims(d) {
    StorageManager::Config cfg;
    cfg.dim = d;
    cfg.metric = 1; // Cosine
    storage.initialize("nebula_data", cfg);

    // Load existing records
    auto dist = getDistFn("cosine");
    uint64_t count = storage.get_record_count();
    std::cout << "Loading " << count << " records from disk..." << std::endl;
    
    for (uint64_t i = 0; i < count; i++) {
        try {
            std::vector<float> emb;
            std::string rawMeta;
            storage.get_record(i, emb, rawMeta);

            // Split meta|cat
            size_t sep = rawMeta.find('|');
            std::string meta = (sep != std::string::npos) ? rawMeta.substr(0, sep) : rawMeta;
            std::string cat = (sep != std::string::npos) ? rawMeta.substr(sep + 1) : "unknown";

            VectorItem v{(int)i, meta, cat, emb};
            store[v.id] = v;
            bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
            if (v.id >= nextId) nextId = v.id + 1;
        } catch (...) {
            std::cerr << "Failed to load record " << i << std::endl;
        }
    }
}

int VectorDB::insert(const std::string& meta, const std::string& cat,
           const std::vector<float>& emb, DistFn dist)
{
    std::lock_guard<std::mutex> lk(mu);
    
    // 1. Persist to Disk
    uint64_t diskId = storage.insert_record(emb, meta + "|" + cat);

    // 2. Update In-Memory Indices
    VectorItem v{(int)diskId, meta, cat, emb};
    store[v.id] = v;
    bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
    
    if (v.id >= nextId) nextId = v.id + 1;
    return v.id;
}

bool VectorDB::remove(int id) {
    std::lock_guard<std::mutex> lk(mu);
    if (!store.count(id)) return false;
    store.erase(id); bf.remove(id); hnsw.remove(id);
    std::vector<VectorItem> rem;
    for (auto& [i, v] : store) rem.push_back(v);
    kdt.rebuild(rem);
    return true;
}

VectorDB::SearchOut VectorDB::search(const std::vector<float>& q, int k,
                 const std::string& metric, const std::string& algo)
{
    std::lock_guard<std::mutex> lk(mu);
    auto dfn = getDistFn(metric);
    auto t0  = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<float,int>> raw;
    if      (algo == "bruteforce") raw = bf.knn(q, k, dfn);
    else if (algo == "kdtree")     raw = kdt.knn(q, k, dfn);
    else                           raw = hnsw.knn(q, k, 50, dfn);

    long long us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();

    SearchOut out; out.us = us; out.algo = algo; out.metric = metric;
    for (auto& [d, id] : raw)
        if (store.count(id))
            out.hits.push_back({id, store[id].metadata, store[id].category, store[id].emb, d});
    return out;
}

VectorDB::BenchOut VectorDB::benchmark(const std::vector<float>& q, int k, const std::string& metric) {
    std::lock_guard<std::mutex> lk(mu);
    auto dfn  = getDistFn(metric);
    auto time = [&](auto fn) -> long long {
        auto t = std::chrono::high_resolution_clock::now();
        fn();
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t).count();
    };
    return {
        time([&]{ bf.knn(q, k, dfn); }),
        time([&]{ kdt.knn(q, k, dfn); }),
        time([&]{ hnsw.knn(q, k, 50, dfn); }),
        (int)store.size()
    };
}

std::vector<VectorItem> VectorDB::all() {
    std::lock_guard<std::mutex> lk(mu);
    std::vector<VectorItem> r;
    for (auto& [id, v] : store) r.push_back(v);
    return r;
}

HNSW::GraphInfo VectorDB::hnswInfo() {
    std::lock_guard<std::mutex> lk(mu);
    return hnsw.getInfo();
}

size_t VectorDB::size() {
    std::lock_guard<std::mutex> lk(mu);
    return store.size();
}
