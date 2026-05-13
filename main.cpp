#include "httplib.h"
#include <iostream>
#include <vector>
#include <string>
#include <mutex>

#include "Core/VectorDB.h"
#include "Core/DocumentDB.h"
#include "Ollama/OllamaClient.h"
#include "Utils/JsonHelpers.h"
#include "Utils/TextChunker.h"

static const int DIMS = 16; 

void cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin",  "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

void loadDemo(VectorDB& db) {
    auto dist = getDistFn("cosine");
    db.insert("Linked List: nodes connected by pointers", "cs",
        {0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, dist);
    db.insert("Binary Search Tree: O(log n) search and insert", "cs",
        {0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, dist);
    db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
        {0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}, dist);
    db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
        {0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}, dist);
    db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
        {0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}, dist);
    db.insert("Calculus: derivatives integrals and limits", "math",
        {0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, dist);
    db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
        {0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}, dist);
    db.insert("Probability: distributions random variables Bayes theorem", "math",
        {0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}, dist);
    db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
        {0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}, dist);
    db.insert("Combinatorics: permutations combinations generating functions", "math",
        {0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}, dist);
    db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
        {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, dist);
    db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
        {0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}, dist);
    db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
        {0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}, dist);
    db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
        {0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}, dist);
    db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
        {0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}, dist);
    db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
        {0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, dist);
    db.insert("Football: tackles touchdowns field goals and strategy", "sports",
        {0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}, dist);
    db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
        {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}, dist);
    db.insert("Chess: openings endgames tactics strategic board game", "sports",
        {0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}, dist);
    db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
        {0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}, dist);
}

int main() {
    VectorDB   db(DIMS);
    DocumentDB docDB;
    OllamaClient ollama;

    loadDemo(db);

    bool ollamaUp = ollama.isAvailable();
    std::cout << "=== NebulaDB Engine (Refactored) ===" << std::endl;
    std::cout << "Listening on http://localhost:8080" << std::endl;
    std::cout << db.size() << " demo vectors | " << DIMS << " dims" << std::endl;
    std::cout << "Ollama: " << (ollamaUp ? "ONLINE" : "OFFLINE") << std::endl;

    httplib::Server svr;

    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        cors(res); res.status = 204;
    });

    svr.Get("/items", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::string o = "[";
        auto all = db.all();
        for (size_t i = 0; i < all.size(); i++) {
            if (i) o += ",";
            o += "{\"id\":" + std::to_string(all[i].id) +
                 ",\"metadata\":" + jS(all[i].metadata) +
                 ",\"category\":" + jS(all[i].category) +
                 ",\"embedding\":" + jVec(all[i].emb) + "}";
        }
        res.set_content(o + "]", "application/json");
    });

    svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto q = parseVec(req.get_param_value("v"));
        int k = 5;
        try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto algo   = req.get_param_value("algo");   if (algo.empty())   algo = "hnsw";

        auto out = db.search(q, k, metric, algo);
        std::string o = "{\"latencyUs\":" + std::to_string(out.us) + ",\"results\":[";
        for (size_t i = 0; i < out.hits.size(); i++) {
            if (i) o += ",";
            o += "{\"id\":" + std::to_string(out.hits[i].id) +
                 ",\"metadata\":" + jS(out.hits[i].meta) +
                 ",\"category\":" + jS(out.hits[i].cat) +
                 ",\"distance\":" + std::to_string(out.hits[i].dist) + "}";
        }
        res.set_content(o + "]}", "application/json");
    });

    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        std::string meta, cat; std::vector<float> emb;
        if (!parseBody(req.body, meta, cat, emb)) {
            res.status = 400; return;
        }
        int id = db.insert(meta, cat, emb, getDistFn("cosine"));
        res.set_content("{\"id\":" + std::to_string(id) + "}", "application/json");
    });

    svr.Delete("/delete/:id", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id = std::stoi(req.path_params.at("id"));
        db.remove(id);
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/benchmark", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto q = parseVec(req.get_param_value("v"));
        int k = 5; try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto m = req.get_param_value("metric"); if (m.empty()) m = "cosine";
        auto b = db.benchmark(q, k, m);
        res.set_content("{\"bruteforceUs\":" + std::to_string(b.bfUs) +
                        ",\"kdtreeUs\":" + std::to_string(b.kdUs) +
                        ",\"hnswUs\":" + std::to_string(b.hnswUs) +
                        ",\"n\":" + std::to_string(b.n) + "}", "application/json");
    });

    svr.Get("/hnsw-info", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto gi = db.hnswInfo();
        std::string o = "{\"topLayer\":" + std::to_string(gi.topLayer) + ",\"nodesPerLayer\":[";
        for (size_t i = 0; i < gi.nodesPerLayer.size(); i++) {
            if (i) o += ","; o += std::to_string(gi.nodesPerLayer[i]);
        }
        o += "],\"edgesPerLayer\":[";
        for (size_t i = 0; i < gi.edgesPerLayer.size(); i++) {
            if (i) o += ","; o += std::to_string(gi.edgesPerLayer[i]);
        }
        res.set_content(o + "]}", "application/json");
    });

    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        bool up = ollama.isAvailable();
        res.set_content("{\"ollamaAvailable\":" + std::string(up?"true":"false") +
                        ",\"persistence\":true" +
                        ",\"embedModel\":" + jS(ollama.embedModel) +
                        ",\"genModel\":" + jS(ollama.genModel) +
                        ",\"docCount\":" + std::to_string(docDB.size()) +
                        ",\"docDims\":" + std::to_string(docDB.getDims()) + "}", "application/json");
    });

    svr.Post("/doc/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        std::string title = extractStr(req.body, "title");
        std::string text  = extractStr(req.body, "text");
        if (title.empty() || text.empty()) { res.status=400; return; }

        auto chunks = chunkText(text);
        int count = 0;
        for (auto& ch : chunks) {
            auto emb = ollama.embed(ch);
            if (!emb.empty()) { docDB.insert(title, ch, emb); count++; }
        }
        res.set_content("{\"chunks\":" + std::to_string(count) +
                        ",\"dims\":" + std::to_string(docDB.getDims()) + "}", "application/json");
    });

    svr.Get("/doc/list", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto all = docDB.all();
        std::string o = "[";
        for (size_t i = 0; i < all.size(); i++) {
            if (i) o += ",";
            std::string prev = all[i].text.substr(0, 120) + "...";
            int words = 0; std::stringstream ss(all[i].text); std::string w; while(ss>>w) words++;
            o += "{\"id\":" + std::to_string(all[i].id) +
                 ",\"title\":" + jS(all[i].title) +
                 ",\"preview\":" + jS(prev) +
                 ",\"words\":" + std::to_string(words) + "}";
        }
        res.set_content(o + "]", "application/json");
    });

    svr.Delete("/doc/delete/:id", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id = std::stoi(req.path_params.at("id"));
        docDB.remove(id);
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Post("/doc/ask", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        std::string q = extractStr(req.body, "question");
        int k = extractInt(req.body, "k", 3);
        auto qEmb = ollama.embed(q);
        if (qEmb.empty()) { res.set_content("{\"error\":\"Ollama error\"}", "application/json"); return; }

        auto hits = docDB.search(qEmb, k);
        std::string context = "";
        std::string ctxJson = "[";
        for (size_t i = 0; i < hits.size(); i++) {
            context += hits[i].second.text + "\n\n";
            if (i) ctxJson += ",";
            ctxJson += "{\"title\":" + jS(hits[i].second.title) + "}";
        }
        ctxJson += "]";

        std::string prompt = "Context:\n" + context + "\n\nQuestion: " + q + 
                             "\n\nAnswer using only the context above. If not in context, say you don't know.";
        std::string answer = ollama.generate(prompt);

        res.set_content("{\"answer\":" + jS(answer) + ",\"contexts\":" + ctxJson + "}", "application/json");
    });

    svr.listen("0.0.0.0", 8080);
    return 0;
}
