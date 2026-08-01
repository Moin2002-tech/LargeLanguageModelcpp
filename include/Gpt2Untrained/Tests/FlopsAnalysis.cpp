//
// Created by moinshaikh on 7/30/26.
//

#include <torch/torch.h>
#include <torch/cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <GPT2LargeLanguageModel/util.hpp>
#include <GPT2LargeLanguageModel/Gpt2Model.hpp>
#include <doctest.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>

/*
 * FLOPs (Floating Point Operations) measure the computational complexity of neural network models.
 *
 * This analysis computes FLOPS analytically for GPT-2 models of various sizes.
 *
 * A single multiply-accumulate (MAC) operation counts as 2 FLOPS (one multiply + one add).
 *
 * For a GPT-2 model, the total FLOPS for one forward pass with batch_size B,
 * sequence length C, embedding dimension D, vocabulary size V, and L layers:
 *
 * Per transformer layer
 *
 * 1. Multi-Head Attention (assuming d_model = d_k * h, where d_k = D/H):
 *    QKV projection:   3 × (2 × B × C × D × D) = 6 × B × C × D²
 *                       (three separate linear layers: Wq, Wk, Wv)
 *    Q × K^T scores:   2 × B × H × C × D/H × C = 2 × B × C² × D
 *    scores × V:       2 × B × H × C × C × D/H = 2 × B × C² × D
 *    Output projection: 2 × B × C × D × D = 2 × B × C × D²
 *                       (out_proj linear layer)
 *    Total attention:  8 × B × C × D² + 4 × B × C² × D
 *
 * 2. Feed-Forward Network:
 *    First linear (D → 4D):  2 × B × C × D × 4D = 8 × B × C × D²
 *    Second linear (4D → D): 2 × B × C × 4D × D = 8 × B × C × D²
 *    Total FFN: 16 × B × C × D²
 *
 *    Total per layer: 24 × B × C × D² + 4 × B × C² × D
 *
 * Output head
 *    LM head: 2 × B × C × D × V
 *
 * Total FLOPS
 *    Total = L × (24 × B × C × D² + 4 × B × C² × D) + 2 × B × C × D × V
 *
 * Notes:
 * - LayerNorm, dropout, GELU, and bias additions contribute negligible FLOPS
 *   and are omitted for this analysis.
 * - The token embedding and position embedding are lookup tables with no FLOPS.
 * - Weight tying is handled (output head weight is shared with token embedding).
 */

/// Compute total FLOPS for one forward pass of a GPT-2 model.
/// All parameters are derived from the config struct.
double compute_flops(const config& cfg, int batch_size = 2, int context_length = 1024)
{
    double B = static_cast<double>(batch_size);
    double C = static_cast<double>(context_length);
    double D = static_cast<double>(cfg.emb_dim);
    double L = static_cast<double>(cfg.n_layer);
    double V = static_cast<double>(cfg.vocab_size);

    // Per transformer layer
    // Attention: QKV projections + score computation + output projection
    double attention_flops = 8.0 * B * C * D * D + 4.0 * B * C * C * D;
    // Feed forward: two linear layers (D → 4D → D)
    double ffn_flops = 16.0 * B * C * D * D;
    double per_layer_flops = attention_flops + ffn_flops;

    // All layers
    double total_layer_flops = L * per_layer_flops;

    //  Output head (LM head)
    double output_head_flops = 2.0 * B * C * D * V;

    return total_layer_flops + output_head_flops;
}

/// Compute the number of trainable parameters for a GPT-2 model.
/// Accounts for weight tying between the token embedding and output head.
long long compute_params(const config& cfg)
{
    long long D = cfg.emb_dim;
    long long L = cfg.n_layer;
    long long V = cfg.vocab_size;
    long long C = cfg.context_length;
    bool bias = cfg.qkv_bias;

    // Token embedding: V × D
    long long tok_embed = V * D;

    // Position embedding: C × D
    long long pos_embed = C * D;

    // Per transformer layer:
    //   W_query, W_keys, W_value: 3 × (D × D) weight + bias (if enabled)
    //   out_proj: D × D weight + bias
    //   FFN first linear: D × 4D weight + bias
    //   FFN second linear: 4D × D weight + bias
    //   Two LayerNorms: 2 × D (weight + bias) × 2
    long long per_layer = 0;

    // QKV projections (3 separate linear layers: W_query, W_keys, W_value)
    per_layer += 3LL * D * D;              // weights
    if (bias) per_layer += 3LL * D;        // biases

    // Attention output projection
    per_layer += (long long)D * D;         // weight
    if (bias) per_layer += D;              // bias

    // FFN first linear: D → 4D
    per_layer += 4LL * D * D;              // weight
    per_layer += 4LL * D;                  // bias (always present in nn::Linear)

    // FFN second linear: 4D → D
    per_layer += 4LL * D * D;              // weight
    per_layer += D;                        // bias (always present in nn::Linear)

    // Two LayerNorms: each has weight and bias, both of size D
    per_layer += 4LL * D;                  // 2 layer norms × 2 params each × D

    // Final LayerNorm: weight + bias
    long long final_ln = 2LL * D;

    // Output head (LM head): bias only (weight is shared with token embedding via weight tying)
    long long output_head = V;             // bias only

    // Bias terms in nn::Linear are always present (default bias=true in torch::nn::Linear)
    // So the bias counting for FFN layers is already included above.

    return tok_embed + pos_embed + L * per_layer + final_ln + output_head;
}

TEST_CASE("flopsAnalysis")
{
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
   // torch::Device device = torch::kCPU;

    // Base configuration (shared across all model sizes)
    config base_config;
    base_config.vocab_size = 50257;
    base_config.context_length = 1024;
    base_config.drop_rate = 0.0;
    base_config.qkv_bias = false;

    // Model configurations
    // (display_name, config_lookup_name, expected_param_count_label)
    std::vector<std::tuple<std::string, std::string, std::string>> model_infos = {
        {"gpt2-small (124M)",  "gpt2-small",   "124M"},
        {"gpt2-medium (355M)", "gpt2-medium",  "355M"},
        {"gpt2-large (774M)",  "gpt2-large",   "774M"},
        {"gpt2-xl (1558M)",    "gpt2-xl",      "1558M"}
    };

    int batch_size = 2;
    int context_length = 1024;

    std::cout << "\n";

    std::cout << "                 GPT-2 FLOPS Analysis (C++)" << "\n";

    std::cout << " Batch size: " << batch_size
              << " | Context length: " << context_length
              << " | Vocab size: " << base_config.vocab_size << "\n";
    std::cout << "\n";

    for (const auto& info : model_infos)
    {
        const std::string& display_name = std::get<0>(info);
        const std::string& config_name  = std::get<1>(info);

        // Get model-specific config (emb_dim, n_layer, n_heads)
        config cfg = get_gpt2_config(config_name);

        // Override with base config values
        cfg.vocab_size     = base_config.vocab_size;
        cfg.context_length = base_config.context_length;
        cfg.drop_rate      = base_config.drop_rate;
        cfg.qkv_bias       = base_config.qkv_bias;

        // Compute FLOPS
        double flops = compute_flops(cfg, batch_size, context_length);

        // Compute parameters
        long long params = compute_params(cfg);


        std::cout << std::left << std::setw(22) << display_name
                  << ": FLOPS = " << std::scientific << std::setprecision(2) << flops
                  << " | Params = " << std::fixed << std::setprecision(0)
                  << (params / 1e6) << "M" << std::endl;
    }

    std::cout << "\n";

    std::cout << "Each FLOPS value represents one forward pass." <<  "\n";
    std::cout << "Training requires ~3x the forward-pass FLOPS" << "\n";
    std::cout << "(forward + backward + weight update)." << "\n";

    std::cout <<  "\n";

    // GPU-aware benchmark: find maximum batch size using exponential search
    // then narrow down with binary search.
    // Strategy: start with batch_size=1, double until failure, then binary search.
    for (const auto & info : model_infos)
    {
        const std::string & display_name = std::get<0>(info);
        const std::string & config_name  = std::get<1>(info);

        config cfg = get_gpt2_config(config_name);
        cfg.vocab_size     = base_config.vocab_size;
        cfg.context_length = base_config.context_length;
        cfg.drop_rate      = base_config.drop_rate;
        cfg.qkv_bias       = base_config.qkv_bias;

        // Ensure the GPU allocator is fully released before starting this model.
        // If previous-model tensors are still referenced, memory_allocated() stays
        // high here — telling us there's a retention leak rather than a HW limit.
        if (device.is_cuda()) {
            c10::cuda::CUDACachingAllocator::emptyCache();
            torch::cuda::synchronize();

            // Diagnostic only — never let it crash the benchmark.
            try {
                int device_id = c10::cuda::current_device();  // actual device index

                c10::cuda::CUDACachingAllocator::DeviceStats stats =
                    c10::cuda::CUDACachingAllocator::getDeviceStats(device_id);

                int64_t allocated_bytes = stats.allocated_bytes[0].current;
                int64_t cached_bytes    = stats.reserved_bytes[0].current;

                std::cout << "    [CUDA allocated before model: "
                          << allocated_bytes << " bytes | "
                          << cached_bytes << " bytes cached]\n";
            } catch (const std::exception&) {
                // ignore — diagnostic only
            }
        }

        std::cout << "\nProcessing " << display_name << "\n";

        int max_batch_size = -1;
        int upper_bound = 1;

        // Phase 1: Exponential search to find an upper bound
        // Start from batch_size=1 and double until it fails
        {
            bool found_upper = false;
            for (int bs = 1; bs <= 4096 && !found_upper; bs *= 2)
            {
                try
                {
                    // Create input tensor: [batch_size, context_length]
                    // Must explicitly specify dtype=Long since embedding layers require Long indices
                    auto input_tensor = torch::randint(
                        0, cfg.vocab_size,
                        {bs, cfg.context_length},
                        torch::TensorOptions().device(device).dtype(torch::kLong)
                    );

                    // Create model
                    Gpt2 model = Gpt2(cfg);
                    model->eval();

                    // Cast to bfloat16 on CPU FIRST, then move to GPU.
                    // model->to() is NOT in-place: it keeps the original FP32 weights
                    // alive while allocating the bf16 copies. Casting on GPU would
                    // need FP32 + bf16 simultaneously (e.g. gpt2-large = 3.1 + 1.55 GB),
                    // which exceeds the 3.62 GiB card. Casting on CPU avoids that.
                    try {
                        model->to(torch::kBFloat16);
                    } catch (const std::exception&) {
                        std::cout << "  (bfloat16 not supported, using float32)\n";
                    }
                    // Move the (half-size) weights to the GPU
                    model->to(device);
                    // Diagnose the effective dtype — helps detect silent FP32 fallback kernels
                    if (device.is_cuda() && bs == 1) {
                        std::cout << "    [dtype after cast: "
                                  << model->parameters()[0].scalar_type() << "]\n";
                    }

                    // Forward pass (no gradient tracking)
                    {
                        torch::NoGradGuard no_grad;
                        auto output = model->forward(input_tensor);
                        // Synchronize to ensure the CUDA kernel actually ran
                        if (device.is_cuda()) {
                            torch::cuda::synchronize();
                        }
                    }

                    // Compute FLOPS analytically
                    double flops = compute_flops(cfg, bs, cfg.context_length);
                    std::cout << "  Batch size " << bs << ": "
                              << std::scientific << std::setprecision(2) << flops
                              << " FLOPS [OK]\n";

                    max_batch_size = bs;
                    upper_bound = bs * 2;
                }
                catch (const std::exception & e)
                {
                    std::string what = e.what();
                    // Check for CUDA OOM (or any CUDA error)
                    if (what.find("out of memory") != std::string::npos ||
                        what.find("CUDA") != std::string::npos ||
                        what.find("cuda") != std::string::npos)
                    {
                        upper_bound = bs;
                        found_upper = true;
                        std::cout << "  Batch size " << bs << ": "
                                  << "OOM/CUDA error (upper bound found)\n"
                                  << "    Error: " << what.substr(0, 120) << "...\n";

                        // Release the CUDA caching allocator's retained blocks
                        // so subsequent batch-size attempts get more usable memory.
                        if (device.is_cuda()) {
                            c10::cuda::CUDACachingAllocator::emptyCache();
                        }
                    }
                    else
                    {
                        // Re-throw non-OOM errors
                        std::cout << "  Batch size " << bs << ": "
                                  << "Non-OOM error: " << what.substr(0, 200) << "\n";
                        throw;
                    }
                }
            }
        }

        // Phase 2: Binary search between last successful and upper bound
        if (max_batch_size >= 0 && upper_bound > max_batch_size + 1)
        {
            int low = max_batch_size + 1;
            int high = upper_bound - 1;

            while (low <= high)
            {
                int mid = (low + high) / 2;

                try
                {
                    auto input_tensor = torch::randint(
                        0, cfg.vocab_size,
                        {mid, cfg.context_length},
                        torch::TensorOptions().device(device).dtype(torch::kLong)
                    );

                    Gpt2 model = Gpt2(cfg);
                    model->eval();

                    // Cast on CPU first, then move to GPU (see note in exponential phase)
                    try {
                        model->to(torch::kBFloat16);
                    } catch (const std::exception&) {
                        // falls back to float32
                    }
                    model->to(device);

                    {
                        torch::NoGradGuard no_grad;
                        auto output = model->forward(input_tensor);
                        if (device.is_cuda())
                        {
                            torch::cuda::synchronize();
                        }
                    }

                    double flops = compute_flops(cfg, mid, cfg.context_length);
                    std::cout << "  Batch size " << mid << ": "
                              << std::scientific << std::setprecision(2) << flops
                              << " FLOPS [OK]\n";

                    max_batch_size = mid;
                    low = mid + 1;
                }
                catch (const std::exception & e)
                {
                    std::string what = e.what();
                    if (what.find("out of memory") != std::string::npos ||
                        what.find("CUDA") != std::string::npos ||
                        what.find("cuda") != std::string::npos)
                    {
                        high = mid - 1;

                        // Release cached blocks so the next (smaller) batch attempt
                        // doesn't inherit a nearly-full allocator.
                        if (device.is_cuda()) {
                            c10::cuda::CUDACachingAllocator::emptyCache();
                        }
                    }
                    else
                    {
                        throw;
                    }
                }
            }
        }

        if (max_batch_size > 0)
        {
            double final_flops = compute_flops(cfg, max_batch_size, cfg.context_length);
            long long params = compute_params(cfg);
            std::cout << "  -> Max batch size: " << max_batch_size
                      << " | FLOPS: " << std::scientific << std::setprecision(2) << final_flops
                      << " | Params: " << std::fixed << std::setprecision(0)
                      << (params / 1e6) << "M\n";
        }
        else
        {
            std::cout << "  -> Failed to find any valid batch size!\n";
        }

        // Model complete — release the CUDA caching allocator's retained blocks.
        // Without this, the weights from this model stay cached in the allocator
        // and eat into the GPU budget of the next, larger model (e.g. gpt2-xl).
        if (device.is_cuda()) {
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
    }
}
