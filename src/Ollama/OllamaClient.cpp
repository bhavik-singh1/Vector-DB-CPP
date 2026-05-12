#include "Ollama/OllamaClient.h"

OllamaClient::OllamaClient(const std::string& h, int p)
    : host(h), port(p) {}

std::string OllamaClient::esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if      (c == '"')  o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else                o += c;
    }
    return o;
}

std::vector<float> OllamaClient::parseEmbedding(const std::string& body) {
    size_t p = body.find("\"embedding\"");
    if (p == std::string::npos) return {};
    p = body.find('[', p);
    if (p == std::string::npos) return {};
    size_t e = p + 1, depth = 1;
    while (e < body.size() && depth > 0) {
        if (body[e] == '[') depth++;
        else if (body[e] == ']') depth--;
        e++;
    }
    return parseVec(body.substr(p + 1, e - p - 2));
}

std::string OllamaClient::parseResponse(const std::string& body) {
    return extractStr(body, "response");
}

bool OllamaClient::isAvailable() {
    httplib::Client cli(host, port);
    cli.set_connection_timeout(2, 0);
    auto res = cli.Get("/api/tags");
    return res && res->status == 200;
}

std::vector<float> OllamaClient::embed(const std::string& text) {
    httplib::Client cli(host, port);
    cli.set_connection_timeout(3, 0);
    cli.set_read_timeout(30, 0);
    std::string body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
    auto res = cli.Post("/api/embeddings", body, "application/json");
    if (!res || res->status != 200) return {};
    return parseEmbedding(res->body);
}

std::string OllamaClient::generate(const std::string& prompt) {
    httplib::Client cli(host, port);
    cli.set_connection_timeout(3, 0);
    cli.set_read_timeout(180, 0);
    std::string body = "{\"model\":\"" + genModel + "\","
                       "\"prompt\":\"" + esc(prompt) + "\","
                       "\"stream\":false}";
    auto res = cli.Post("/api/generate", body, "application/json");
    if (!res || res->status != 200)
        return "ERROR: Ollama unavailable. Run: ollama serve";
    return parseResponse(res->body);
}
