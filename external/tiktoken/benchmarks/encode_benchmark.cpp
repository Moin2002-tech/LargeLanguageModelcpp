#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../src/tiktoken.hpp"

// Minimal benchmark:
//   ./tiktoken_benchmark --bpe <path-to-.tiktoken> --file <textfile> --iters 1000 [--progress]
// If --bpe is omitted, a minimal demo encoding is used.

static tiktoken::EncodingDefinition make_minimal_demo_def() {
    using namespace tiktoken;
    std::vector<std::pair<U8Vec, Rank>> ranks;
    ranks.reserve(256);
    for (int b = 0; b < 256; ++b) ranks.push_back({U8Vec{static_cast<U8>(b)}, static_cast<Rank>(b)});
    std::vector<std::pair<std::string, Rank>> specials = {{"<|endoftext|>", 256}};
    EncodingDefinition def;
    def.name = "minimal_demo";
    def.pat_str = R"('(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s)";
    def.mergeable_ranks = std::move(ranks);
    def.special_tokens = std::move(specials);
    def.explicit_n_vocab = 257;
    return def;
}

static tiktoken::EncodingDefinition load_bpe_def(const std::string &path) {
    using namespace tiktoken;
    auto ranks = load_tiktoken_bpe_from_file(path);
    std::vector<std::pair<std::string, Rank>> specials = {
        {"<|endoftext|>", 100257},
        {"<|fim_prefix|>", 100258},
        {"<|fim_middle|>", 100259},
        {"<|fim_suffix|>", 100260},
        {"<|endofprompt|>", 100276},
    };
    EncodingDefinition def;
    def.name = "cl100k_base";
    def.pat_str = R"('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s)";
    def.mergeable_ranks = std::move(ranks);
    def.special_tokens = std::move(specials);
    return def;
}

static std::string read_file_text(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open: " + path);
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return s;
}

int main(int argc, char **argv) {
    using namespace tiktoken;
    std::string bpe_path;
    std::string text_file;
    int iters = 1000;
    bool progress = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bpe" && i + 1 < argc) bpe_path = argv[++i];
        else if (a == "--file" && i + 1 < argc) text_file = argv[++i];
        else if (a == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
        else if (a == "--progress") progress = true;
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage: ./tiktoken_benchmark --bpe <file> --file <text> --iters N [--progress]\n";
            return 0;
        }
    }

    std::shared_ptr<Encoding> enc;
    if (!bpe_path.empty()) {
        auto def = load_bpe_def(bpe_path);
        register_encoding(def);
        enc = get_encoding(def.name);
    } else {
        auto def = make_minimal_demo_def();
        register_encoding(def);
        enc = get_encoding(def.name);
    }

    std::string text = text_file.empty() ? std::string("hello world\n") : read_file_text(text_file);

    // Warm-up
    auto warm = enc->encode(text);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::size_t total_tokens = 0;
    for (int i = 0; i < iters; ++i) {
        auto toks = enc->encode(text);
        total_tokens += toks.size();
        if (progress && (i % (iters/10 + 1) == 0)) std::cerr << ".";
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Iters: " << iters << ", tokens: " << total_tokens
              << ", time_ms: " << ms << ", tok/s: " << (total_tokens / (ms/1000.0)) << "\n";
    return 0;
}
