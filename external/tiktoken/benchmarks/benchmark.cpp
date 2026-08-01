#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../src/tiktoken.hpp"

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

// ============================================================================
// Utilities
// ============================================================================

std::string generate_random_text(std::size_t size, unsigned seed = 42) {
    static const char* words[] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "hello", "world", "this", "is", "a", "test", "of", "tokenization",
        "performance", "benchmark", "comparing", "implementations",
        "artificial", "intelligence", "machine", "learning", "neural",
        "network", "deep", "transformer", "attention", "mechanism",
        "encoder", "decoder", "embedding", "vocabulary", "token",
        "Python", "CPlusPlus", "Rust", "JavaScript", "programming", "language",
        "function", "class", "method", "variable", "constant", "loop",
        "conditional", "recursion", "iteration", "algorithm", "data",
        "structure", "array", "vector", "map", "set", "queue", "stack",
        "computer", "science", "software", "hardware", "memory", "cache",
        "processor", "thread", "parallel", "concurrent", "async", "await",
    };
    constexpr std::size_t num_words = sizeof(words) / sizeof(words[0]);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> word_dist(0, num_words - 1);
    std::uniform_int_distribution<int> space_dist(0, 10);

    std::string result;
    result.reserve(size + 100);

    while (result.size() < size) {
        result += words[word_dist(rng)];
        int sep = space_dist(rng);
        if (sep < 6) result += ' ';
        else if (sep < 8) result += '\n';
        else if (sep < 9) result += ", ";
        else result += ". ";
    }

    if (result.size() > size) {
        result.resize(size);
        auto last_space = result.rfind(' ');
        if (last_space != std::string::npos && last_space > size / 2) {
            result.resize(last_space + 1);
        }
    }
    return result;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file: " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

struct BenchmarkResult {
    std::string name;
    std::size_t text_size;
    std::size_t num_tokens;
    double time_ms;
    double tokens_per_sec;
    double mb_per_sec;
};

void print_header() {
    std::cout << std::left << std::setw(30) << "Benchmark"
              << std::setw(12) << "Size"
              << std::setw(12) << "Tokens"
              << std::setw(12) << "Time(ms)"
              << std::setw(15) << "Tokens/sec"
              << std::setw(12) << "MB/sec" << "\n";
    std::cout << std::string(93, '-') << "\n";
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::left << std::setw(30) << r.name
              << std::setw(12) << r.text_size
              << std::setw(12) << r.num_tokens
              << std::fixed << std::setprecision(2) << std::setw(12) << r.time_ms
              << std::setprecision(0) << std::setw(15) << r.tokens_per_sec
              << std::setprecision(2) << std::setw(12) << r.mb_per_sec << "\n";
}

// ============================================================================
// Benchmark Modes
// ============================================================================

void run_throughput_benchmark(const tiktoken::Encoding& enc, int iterations) {
    std::cout << "\n=== Throughput Benchmark (Single Thread) ===\n";
    print_header();

    std::vector<std::size_t> sizes = {1024, 4096, 16384, 65536, 262144, 1048576};
    
    for (auto size : sizes) {
        std::string text = generate_random_text(size);
        
        // Warmup
        enc.encode_ordinary(text);

        auto start = Clock::now();
        std::size_t total_tokens = 0;

        for (int i = 0; i < iterations; ++i) {
            auto tokens = enc.encode_ordinary(text);
            total_tokens += tokens.size();
        }

        auto end = Clock::now();
        double time_ms = std::chrono::duration_cast<Ms>(end - start).count();
        
        BenchmarkResult r;
        r.name = "single_" + std::to_string(size/1024) + "KB";
        r.text_size = size;
        r.num_tokens = total_tokens / iterations;
        r.time_ms = time_ms;
        r.tokens_per_sec = (total_tokens / time_ms) * 1000.0;
        r.mb_per_sec = (size * iterations / (1024.0 * 1024.0)) / (time_ms / 1000.0);
        
        print_result(r);
    }
}

void run_parallel_benchmark(const tiktoken::Encoding& enc, int iterations) {
    std::cout << "\n=== Parallel Benchmark (Single Large Text) ===\n";
    print_header();

    std::vector<std::size_t> sizes = {65536, 262144, 1048576, 4194304};
    
    for (auto size : sizes) {
        std::string text = generate_random_text(size);
        
        // Warmup
        enc.encode_ordinary_parallel(text);

        auto start = Clock::now();
        std::size_t total_tokens = 0;

        for (int i = 0; i < iterations; ++i) {
            auto tokens = enc.encode_ordinary_parallel(text);
            total_tokens += tokens.size();
        }

        auto end = Clock::now();
        double time_ms = std::chrono::duration_cast<Ms>(end - start).count();
        
        BenchmarkResult r;
        r.name = "parallel_" + std::to_string(size/1024) + "KB";
        r.text_size = size;
        r.num_tokens = total_tokens / iterations;
        r.time_ms = time_ms;
        r.tokens_per_sec = (total_tokens / time_ms) * 1000.0;
        r.mb_per_sec = (size * iterations / (1024.0 * 1024.0)) / (time_ms / 1000.0);
        
        print_result(r);
    }
}

void run_batch_benchmark(const tiktoken::Encoding& enc, int iterations) {
    std::cout << "\n=== Batch Benchmark (Multiple Texts) ===\n";
    print_header();

    struct Scenario {
        std::string name;
        std::size_t doc_size; // approximate chars
        int num_docs;
    };

    std::vector<Scenario> scenarios = {
        {"1k_tok_10_doc", 4000, 10},
        {"1k_tok_10k_doc", 4000, 10000},
        {"100_tok_10k_doc", 400, 10000}
    };

    int hw_threads = std::thread::hardware_concurrency();
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 32};

    for (const auto& sc : scenarios) {
        std::cout << "--- Scenario: " << sc.name << " ---\n";
        
        std::vector<std::string> batch_texts;
        std::size_t total_size = 0;
        batch_texts.reserve(sc.num_docs);
        
        // Generate texts once
        std::string base_text = generate_random_text(sc.doc_size);
        for (int i = 0; i < sc.num_docs; ++i) {
            batch_texts.push_back(base_text); // Copy is fine for setup
            total_size += base_text.size();
        }

        for (int threads : thread_counts) {
            // Warmup
            enc.encode_ordinary_batch(batch_texts, threads);

            auto start = Clock::now();
            std::size_t total_tokens = 0;

            // Adjust iterations for large batches to save time
            int current_iters = iterations;
            if (sc.num_docs > 1000) current_iters = std::max(1, iterations / 5);

            for (int i = 0; i < current_iters; ++i) {
                auto results = enc.encode_ordinary_batch(batch_texts, threads);
                for (const auto& res : results) total_tokens += res.size();
            }

            auto end = Clock::now();
            double time_ms = std::chrono::duration_cast<Ms>(end - start).count();
            
            BenchmarkResult r;
            r.name = sc.name + "_" + std::to_string(threads) + "t";
            r.text_size = total_size;
            r.num_tokens = total_tokens / current_iters;
            r.time_ms = time_ms / current_iters; // Average time per iter
            r.tokens_per_sec = (total_tokens / time_ms) * 1000.0;
            r.mb_per_sec = (total_size * current_iters / (1024.0 * 1024.0)) / (time_ms / 1000.0);
            
            print_result(r);
        }
    }
}

void run_file_benchmark(const tiktoken::Encoding& enc, const std::string& path, int iterations) {
    std::cout << "\n=== File Benchmark ===\n";
    print_header();

    std::string text = read_file(path);
    
    // Warmup
    enc.encode_ordinary(text);

    auto start = Clock::now();
    std::size_t total_tokens = 0;

    for (int i = 0; i < iterations; ++i) {
        auto tokens = enc.encode_ordinary(text); // Will auto-select parallel if large enough
        total_tokens += tokens.size();
    }

    auto end = Clock::now();
    double time_ms = std::chrono::duration_cast<Ms>(end - start).count();
    
    BenchmarkResult r;
    r.name = "file_" + path;
    r.text_size = text.size();
    r.num_tokens = total_tokens / iterations;
    r.time_ms = time_ms;
    r.tokens_per_sec = (total_tokens / time_ms) * 1000.0;
    r.mb_per_sec = (text.size() * iterations / (1024.0 * 1024.0)) / (time_ms / 1000.0);
    
    print_result(r);
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --bpe <path>       Path to .tiktoken BPE file (required)\n"
              << "  --mode <mode>      Benchmark mode: all, throughput, parallel, batch, file (default: all)\n"
              << "  --file <path>      Input file for 'file' mode\n"
              << "  --iters <n>        Number of iterations (default: 50)\n"
              << "  --help             Show this help\n";
}

int main(int argc, char** argv) {
    std::string bpe_path;
    std::string mode = "all";
    std::string input_file;
    int iterations = 50;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--bpe" && i + 1 < argc) bpe_path = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--file" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "--iters" && i + 1 < argc) iterations = std::stoi(argv[++i]);
        else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (bpe_path.empty()) {
        std::cerr << "Error: --bpe argument is required.\n";
        print_usage(argv[0]);
        return 1;
    }

    try {
        // Load encoding
        auto ranks = tiktoken::load_tiktoken_bpe_from_file(bpe_path);
        std::vector<std::pair<std::string, tiktoken::Rank>> specials = {
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
        tiktoken::register_encoding(def);
        auto enc = tiktoken::get_encoding("cl100k_base");

        std::cout << "Loaded encoding: " << enc->name() << " (vocab: " << enc->n_vocab() << ")\n";
        std::cout << "Iterations: " << iterations << "\n";

        if (mode == "all" || mode == "throughput") {
            run_throughput_benchmark(*enc, iterations);
        }
        if (mode == "all" || mode == "parallel") {
            run_parallel_benchmark(*enc, iterations);
        }
        if (mode == "all" || mode == "batch") {
            run_batch_benchmark(*enc, iterations);
        }
        if (mode == "file") {
            if (input_file.empty()) {
                std::cerr << "Error: --file argument required for file mode\n";
                return 1;
            }
            run_file_benchmark(*enc, input_file, iterations);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
