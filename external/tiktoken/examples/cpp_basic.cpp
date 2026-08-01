#include <iostream>
#include <string>
#include <vector>
#include <set>

#include "../src/tiktoken.hpp"

// Usage:
//   1) Download a .tiktoken BPE file locally (e.g., cl100k_base.tiktoken)
//   2) Build this example with CMake
//   3) Run: ./cpp_basic <path-to-.tiktoken> "your text here"
// If no args are provided, a small built-in fallback demo runs with a dummy minimal vocab.

static tiktoken::EncodingDefinition make_minimal_demo_def() {
    using namespace tiktoken;
    // Minimal artificial encoding: bytes map directly to ranks (0..255), no merges.
    std::vector<std::pair<U8Vec, Rank>> ranks;
    ranks.reserve(256);
    for (int b = 0; b < 256; ++b) ranks.push_back({U8Vec{static_cast<U8>(b)}, static_cast<Rank>(b)});

    std::vector<std::pair<std::string, Rank>> specials = {
        {"<|endoftext|>", 256}
    };

    EncodingDefinition def;
    def.name = "minimal_demo";
    def.pat_str = R"('(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s)"; // r50k-like
    def.mergeable_ranks = std::move(ranks);
    def.special_tokens = std::move(specials);
    def.explicit_n_vocab = 257;
    return def;
}

static tiktoken::EncodingDefinition load_cl100k_def(const std::string& tiktoken_path) {
    using namespace tiktoken;
    auto ranks = load_tiktoken_bpe_from_file(tiktoken_path);
    // cl100k special tokens
    std::vector<std::pair<std::string, Rank>> specials = {
        {"<|endoftext|>", 100257},
        {"<|fim_prefix|>", 100258},
        {"<|fim_middle|>", 100259},
        {"<|fim_suffix|>", 100260},
        {"<|endofprompt|>", 100276},
    };
    tiktoken::EncodingDefinition def;
    def.name = "cl100k_base";
        def.pat_str = R"('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s)";
    def.mergeable_ranks = std::move(ranks);
    def.special_tokens = std::move(specials);
    // Note: cl100k_base does not specify explicit_n_vocab in Python; omit here too.
    return def;
}

int main(int argc, char** argv) {
    using namespace tiktoken;

    std::shared_ptr<Encoding> enc;
    std::string text = "hello world";

    try {
        if (argc >= 3) {
            // Load cl100k from file
            auto def = load_cl100k_def(argv[1]);
            register_encoding(def);
            enc = get_encoding("cl100k_base");
            text = argv[2];
        } else {
            // Minimal fallback demo
            auto def = make_minimal_demo_def();
            register_encoding(def);
            enc = get_encoding(def.name);
            text = "hello world";
        }

        std::cout << "Encoding: " << enc->name() << "\n";
        std::cout << "Text: " << text << "\n";

        auto tokens = enc->encode(text);
        std::cout << "Tokens (" << tokens.size() << "): ";
        for (auto t : tokens) std::cout << t << ' ';
        std::cout << "\n";

        auto bytes = enc->decode_bytes(tokens);
        std::string decoded(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        std::cout << "Decoded: " << decoded << "\n";

        auto pair = enc->decode_with_offsets(tokens);
        std::cout << "Offsets: ";
        for (auto off : pair.second) std::cout << off << ' ';
        std::cout << "\n";

        // Unstable encoding demo (allowed_special: endoftext)
        std::set<std::string> allowed = {"<|endoftext|>"};
        auto unstable = enc->encode_with_unstable(text, allowed);
        std::cout << "Stable tokens: ";
        for (auto t : unstable.first) std::cout << t << ' ';
        std::cout << "\nCompletions: [" << unstable.second.size() << "]\n";
        for (const auto& seq : unstable.second) {
            for (auto t : seq) std::cout << t << ' ';
            std::cout << "| ";
        }
        std::cout << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
