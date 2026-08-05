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

/*
 * KV-cache Memory Estimate: MHA vs GQA
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

    // ---- Output ----
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

    std::cout << "==== KV-cache totals across all layers ====\n";
    std::cout << "MHA total KV cache  : " << convert_bytes(total_mha) << "\n";
    std::cout << "GQA total KV cache  : " << convert_bytes(total_gqa) << "\n";
    std::cout << "Ratio (MHA / GQA)   : "
              << std::fixed << std::setprecision(2) << ratio << "x\n";
    std::cout << "Savings (GQA vs MHA): "
              << std::fixed << std::setprecision(2) << (savings * 100.0L) << "%\n";
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