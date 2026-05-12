#pragma once
#include <string>
#include <vector>

struct VectorItem {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> emb;
};
