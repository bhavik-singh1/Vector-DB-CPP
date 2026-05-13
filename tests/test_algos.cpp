#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include "Algorithms/KDTree.h"
#include "Algorithms/HNSW.h"
#include "Algorithms/BruteForce.h"
#include "Utils/DistanceMetrics.h"

#define TEST_PASS "\033[32m[PASS]\033[0m "
#define TEST_FAIL "\033[31m[FAIL]\033[0m "

void test_kdtree() {
    std::cout << "Running KD-Tree Tests..." << std::endl;
    KDTree tree(2);
    DistFn dist = euclidean;

    // Insert simple points
    tree.insert({1, "A", "test", {0.1f, 0.1f}});
    tree.insert({2, "B", "test", {0.9f, 0.9f}});
    tree.insert({3, "C", "test", {0.5f, 0.5f}});

    // Search near A
    auto res = tree.knn({0.15f, 0.15f}, 1, dist);
    assert(!res.empty());
    if (res[0].second == 1) {
        std::cout << TEST_PASS << "KD-Tree nearest neighbor correct." << std::endl;
    } else {
        std::cout << TEST_FAIL << "KD-Tree nearest neighbor incorrect. Got ID: " << res[0].second << std::endl;
    }
}

void test_hnsw_accuracy() {
    std::cout << "\nRunning HNSW Accuracy (Recall) Tests..." << std::endl;
    int dims = 16;
    HNSW hnsw(16, 200);
    BruteForce bf;
    DistFn dist = cosine;

    // Insert 100 random vectors
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < 100; i++) {
        std::vector<float> v(dims);
        for (float& f : v) f = dis(gen);
        VectorItem item = {i, "meta", "cat", v};
        hnsw.insert(item, dist);
        bf.insert(item);
    }

    // Run 20 queries and check recall
    int hits = 0;
    for (int i = 0; i < 20; i++) {
        std::vector<float> q(dims);
        for (float& f : q) f = dis(gen);

        auto res_hnsw = hnsw.knn(q, 1, 50, dist);
        auto res_bf = bf.knn(q, 1, dist);

        if (!res_hnsw.empty() && !res_bf.empty() && res_hnsw[0].second == res_bf[0].second) {
            hits++;
        }
    }

    float recall = (float)hits / 20.0f;
    std::cout << "HNSW Recall@1: " << std::fixed << std::setprecision(2) << recall * 100 << "%" << std::endl;
    
    if (recall >= 0.9f) {
        std::cout << TEST_PASS << "HNSW accuracy within acceptable range." << std::endl;
    } else {
        std::cout << TEST_FAIL << "HNSW accuracy too low!" << std::endl;
    }
}

int main() {
    try {
        test_kdtree();
        test_hnsw_accuracy();
        std::cout << "\nAll tests completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
