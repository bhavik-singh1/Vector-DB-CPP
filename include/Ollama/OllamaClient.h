#pragma once
#include <string>
#include <vector>
#include "httplib.h"
#include "Utils/JsonHelpers.h"

class OllamaClient {
    std::string host;
    int         port;

    std::string esc(const std::string& s);
    std::vector<float> parseEmbedding(const std::string& body);
    std::string parseResponse(const std::string& body);

public:
    std::string embedModel = "nomic-embed-text";
    std::string genModel   = "llama3.2";

    OllamaClient(const std::string& h = "127.0.0.1", int p = 11434);

    bool isAvailable();
    std::vector<float> embed(const std::string& text);
    std::string generate(const std::string& prompt);
};
