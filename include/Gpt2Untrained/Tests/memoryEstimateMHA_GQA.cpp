//
// Created by moinshaikh on 8/5/26.
//

#include <doctest.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * KV-cache Memory & Bandwidth Estimate: MHA vs GQA
 *
 * 1. KV-cache Memory
 *
 * This analysis estimates the KV-cache memory footprint for
 * Multi-Head Attention (MHA) vs Grouped-Query Attention (GQA).
 *
 * For each transformer layer, the KV cache stores K and V tensors of shape:
 *   [batch_size, context_length, n_kv_heads, head_dim]
 *
 * head_dim = ceil(emb_dim / n_heads)
 * n_kv_heads (MHA) = n_heads                    (each query head has its own KV)
 * n_kv_heads (GQA) = n_heads / n_kv_groups      (query heads share KV heads)
 *
 * Per layer bytes = batch * context_length * head_dim * n_kv_heads * 2 * bytes_per_elem
 *                   (the ×2 accounts for both K and V)
 * Total bytes     = per_layer_bytes * n_layers
 *
 * GQA reduces the KV-cache memory by a factor of n_kv_groups compared to MHA.
 *
 * 2. Memory Bandwidth (Decoding)
 *
 * During autoregressive decoding, each newly generated token attends to ALL
 * previous tokens, so the entire KV cache must be read from memory once per
 * generated token. This makes KV-cache traffic the dominant memory-bandwidth
 * consumer during generation.
 *
 * Bytes read per token   = total KV-cache bytes  (same as full cache size)
 * Required bandwidth     = bytes_per_token × tokens_per_second
 *                          (time to decode one token = bytes_per_token / bandwidth)
 *
 * We report:
 *   - Peak memory bandwidth for decoding one token (GB/s)
 *   - Required bandwidth at common target generation rates (10/50/100 tok/s)
 *   - Achievable generation rate on available hardware memory bandwidths
 *     (e.g. Quadro RTX T600 ~160 GB/s, RTX 3060 ~360 GB/s, RTX 4090 ~1008 GB/s)
 */

/// Number of bytes per element for each supported dtype.
const std::map<std::string, int> DTYPE_BYTES = {
    {"fp32", 4},
    {"bf16", 2},
    {"fp16", 2},
    {"fp8",  1},
    {"int8", 1}
};

/// Format a byte count as GB (decimal, 1000³).
std::string convert_bytes(long double n)
{
    long double gb = n / 1000.0L / 1000.0L / 1000.0L;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << gb << " GB";
    return oss.str();
}

/// Format a byte count as MB (decimal, 1000²).
std::string convert_bytes_mb(long double n)
{
    long double mb = n / 1000.0L / 1000.0L;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << mb << " MB";
    return oss.str();
}

/// Compute total KV-cache bytes across all layers.
long double calc_kv_bytes_total(
    int batch,
    int context_length,
    int emb_dim,
    int n_heads,
    int n_kv_heads,
    int n_layers,
    int bytes_per_elem)
{
    long double head_dim = std::ceil(static_cast<long double>(emb_dim) /
                                     static_cast<long double>(n_heads));
    long double per_layer = static_cast<long double>(batch) *
                            static_cast<long double>(context_length) *
                            head_dim *
                            static_cast<long double>(n_kv_heads) *
                            2.0L *
                            static_cast<long double>(bytes_per_elem);
    return per_layer * static_cast<long double>(n_layers);
}

/// Compute peak memory bandwidth required to decode one token (GB/s),
/// assuming the full KV cache must be streamed from memory.
///  - n_bytes   : total KV-cache bytes
///  - time_us   : target decode time per token in microseconds
long double calc_bandwidth_gbps(long double n_bytes, long double time_us)
{
    // GB/s = bytes / 1e9  /  (us / 1e6)  =  bytes × 1e-3 / us
    return n_bytes * 1e-3L / time_us;
}

/// Convert a bandwidth number to a human-readable string.
std::string format_bandwidth(long double gbps)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << gbps << " GB/s";
    return oss.str();
}

TEST_CASE("kvCacheMemoryEstimateMHA_GQA")
{
    // ---- Default configuration (matches Python argparse defaults) ----
    int context_length = 1024;
    int emb_dim        = 768;      // required, example: gpt2
    int n_heads        = 12;       // required
    int n_layers       = 12;       // required
    int n_kv_groups    = 4;        // required
    int batch_size     = 1;
    std::string dtype  = "fp16";

    // ---- Input validation ----
    if (DTYPE_BYTES.find(dtype) == DTYPE_BYTES.end())
    {
        throw std::invalid_argument("Invalid dtype: " + dtype);
    }
    if (n_heads % n_kv_groups != 0)
    {
        throw std::invalid_argument("n_kv_groups must divide n_heads exactly.");
    }

    int bytes_per_elem = DTYPE_BYTES.at(dtype);
    int head_dim       = static_cast<int>(std::ceil(static_cast<double>(emb_dim) /
                                                    static_cast<double>(n_heads)));

    int n_kv_heads_mha = n_heads;
    int n_kv_heads_gqa = n_heads / n_kv_groups;

    long double total_mha = calc_kv_bytes_total(
        batch_size,
        context_length,
        emb_dim,
        n_heads,
        n_kv_heads_mha,
        n_layers,
        bytes_per_elem);

    long double total_gqa = calc_kv_bytes_total(
        batch_size,
        context_length,
        emb_dim,
        n_heads,
        n_kv_heads_gqa,
        n_layers,
        bytes_per_elem);

    long double ratio   = total_mha / total_gqa;
    long double savings = 1.0L - (total_gqa / total_mha);

    // ---- Output: Config ----
    std::cout << "\n==== Config ====\n";
    std::cout << std::left << std::setw(17) << "context_length" << ": " << context_length << "\n";
    std::cout << std::left << std::setw(17) << "emb_dim"        << ": " << emb_dim        << "\n";
    std::cout << std::left << std::setw(17) << "n_heads"        << ": " << n_heads        << "\n";
    std::cout << std::left << std::setw(17) << "n_layers"       << ": " << n_layers       << "\n";
    std::cout << std::left << std::setw(17) << "n_kv_groups"    << ": " << n_kv_groups    << "\n";
    std::cout << std::left << std::setw(17) << "batch_size"     << ": " << batch_size     << "\n";
    std::cout << std::left << std::setw(17) << "dtype"          << ": " << dtype
              << " (" << bytes_per_elem << " Bytes/elem)\n";
    std::cout << std::left << std::setw(17) << "head_dim"       << ": " << head_dim       << "\n";
    std::cout << std::left << std::setw(17) << "GQA n_kv_heads" << ": " << n_kv_heads_gqa << "\n";
    std::cout << "\n";

    // ---- Output: KV-cache totals ----
    std::cout << "==== KV-cache totals across all layers ====\n";
    std::cout << "MHA total KV cache  : " << convert_bytes(total_mha) << "\n";
    std::cout << "GQA total KV cache  : " << convert_bytes(total_gqa) << "\n";
    std::cout << "Ratio (MHA / GQA)   : "
              << std::fixed << std::setprecision(2) << ratio << "x\n";
    std::cout << "Savings (GQA vs MHA): "
              << std::fixed << std::setprecision(2) << (savings * 100.0L) << "%\n";
    std::cout << "\n";

    // ---- Output: Bytes read per token (batch = 1) ----
    // During decoding each token attends to all previously cached tokens,
    // so the entire KV cache is streamed once per generated token.
    std::cout << "==== Memory bandwidth: KV-cache read per generated token ====\n";
    std::cout << "Bytes read / token (MHA) : " << convert_bytes_mb(total_mha) << "\n";
    std::cout << "Bytes read / token (GQA) : " << convert_bytes_mb(total_gqa) << "\n";

    // Time to decode one token if only KV-cache traffic were the bottleneck.
    // 10 ms/token  = 100 tokens/sec
    // 20 ms/token  = 50  tokens/sec
    // 100 ms/token = 10  tokens/sec
    struct TargetRate { long double tokens_per_sec; const char* label; };
    const std::vector<TargetRate> target_rates = {
        { 100.0L, "100 tok/s" },
        {  50.0L, " 50 tok/s" },
        {  10.0L, " 10 tok/s" }
    };

    std::cout << "\n  Required bandwidth to sustain target generation rate:\n";
    std::cout << "  " << std::left << std::setw(14) << "Rate"
              << std::setw(16) << "MHA (GB/s)"
              << "GQA (GB/s)\n";
    for (const auto& rate : target_rates)
    {
        long double us_per_token = 1e6L / rate.tokens_per_sec;
        long double bw_mha = calc_bandwidth_gbps(total_mha, us_per_token);
        long double bw_gqa = calc_bandwidth_gbps(total_gqa, us_per_token);

        std::cout << "  " << std::left << std::setw(14) << rate.label
                  << std::setw(16) << format_bandwidth(bw_mha)
                  << format_bandwidth(bw_gqa) << "\n";
    }

    // ---- Output: Achievable generation rate on common hardware ----
    std::cout << "\n  Achievable generation rate (KV-cache-bound) on common hardware:\n";
    std::cout << "  " << std::left << std::setw(22) << "Device"
              << std::setw(20) << "MHA (tok/s)"
              << "GQA (tok/s)\n";

    // User's hardware: NVIDIA Quadro RTX T600 (4 GB GDDR6, 128-bit bus ~160 GB/s)
    struct DeviceBw { const char* name; long double gbps; };
    const std::vector<DeviceBw> devices = {
        { "Quadro RTX T600 4GB (~160 GB/s)",  160.0L  },
        { "RTX 3060 12GB (~360 GB/s)",        360.0L  },
        { "RTX 4090 24GB (~1008 GB/s)",       1008.0L }
    };

    for (const auto& dev : devices)
    {
        // tokens/s = bandwidth (bytes/s) / bytes_per_token
        long double tok_mha = (dev.gbps * 1e9L) / total_mha;
        long double tok_gqa = (dev.gbps * 1e9L) / total_gqa;

        std::cout << "  " << std::left << std::setw(22) << dev.name
                  << std::setw(20) << (std::ostringstream{} << std::fixed << std::setprecision(0) << tok_mha).str()
                  << (std::ostringstream{} << std::fixed << std::setprecision(0) << tok_gqa).str() << "\n";
    }
    std::cout << std::endl;
}

TEST_CASE("kvCacheMemoryEstimate_inputValidation")
{
    // n_kv_groups must divide n_heads exactly
    bool threw = false;
    try {
        int n_heads     = 12;
        int n_kv_groups = 5;  // invalid
        if (n_heads % n_kv_groups != 0)
        {
            throw std::invalid_argument("n_kv_groups must divide n_heads exactly.");
        }
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw == true);
}

TEST_CASE("kvCacheMemoryEstimate_bandwidthHelper")
{
    // 1 GB (1e9 bytes) streamed in 1 ms = 1e12 B/s = 1000 GB/s
    long double bw = calc_bandwidth_gbps(1e9L, 1e3L);  // = 1000 GB/s
    CHECK(static_cast<double>(bw) == doctest::Approx(1000.0).epsilon(1e-12));
}
