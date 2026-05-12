#include "Core/DocumentDB.h"

DocumentDB::DocumentDB() : hnsw(16, 200) {}

int DocumentDB::insert(const std::string& title, const std::string& text,
           const std::vector<float>& emb)
{
    std::lock_guard<std::mutex> lk(mu);
    if (dims == 0) dims = (int)emb.size();
    DocItem item{nextId++, title, text, emb};
    store[item.id] = item;
    VectorItem vi{item.id, title, "doc", emb};
    hnsw.insert(vi, cosine);
    bf.insert(vi);
    return item.id;
}

std::vector<std::pair<float, DocItem>> DocumentDB::search(
    const std::vector<float>& q, int k, float max_dist)
{
    std::lock_guard<std::mutex> lk(mu);
    if (store.empty()) return {};
    auto raw = (store.size() < 10)
               ? bf.knn(q, k, cosine)
               : hnsw.knn(q, k, 50, cosine);
    std::vector<std::pair<float, DocItem>> out;
    for (auto& [d, id] : raw)
        if (store.count(id) && d <= max_dist) out.push_back({d, store[id]});
    return out;
}

bool DocumentDB::remove(int id) {
    std::lock_guard<std::mutex> lk(mu);
    if (!store.count(id)) return false;
    store.erase(id); hnsw.remove(id); bf.remove(id);
    return true;
}

std::vector<DocItem> DocumentDB::all() {
    std::lock_guard<std::mutex> lk(mu);
    std::vector<DocItem> r;
    for (auto& [id, v] : store) r.push_back(v);
    return r;
}

size_t DocumentDB::size() {
    std::lock_guard<std::mutex> lk(mu);
    return store.size();
}
