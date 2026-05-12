#pragma once
#include <unordered_map>
#include <mutex>
#include "Core/Types.h"
#include "Algorithms/HNSW.h"
#include "Algorithms/BruteForce.h"

struct DocItem {
    int         id;
    std::string title;
    std::string text;
    std::vector<float> emb;
};

class DocumentDB {
    std::unordered_map<int, DocItem> store;
    HNSW       hnsw;
    BruteForce bf;
    std::mutex mu;
    int nextId = 1;
    int dims   = 0;

public:
    DocumentDB();

    int insert(const std::string& title, const std::string& text,
               const std::vector<float>& emb);
    std::vector<std::pair<float, DocItem>> search(
        const std::vector<float>& q, int k, float max_dist = 0.7f);
    bool remove(int id);
    std::vector<DocItem> all();
    size_t size();
    int getDims() { return dims; }
};
