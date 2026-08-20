//
// Created by moinshaikh on 8/10/26.
//


#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include <Gpt2Untrained/Tests/MultiHeadLatent/include/gpt2mhl.hpp>
#include <Gpt2Untrained/dataPreparation.hpp>
#include <doctest.hpp>

// -----------------------------------------------------------------------------
// generateTextSimpleCached
//
// Mirrors the Python generate_text_simple_cached() function:
//   - use_cache=true :  resets the KV cache, pre-fills with the prompt, then
//                       feeds only the newly generated token at each step.
//   - use_cache=false:  refeeds the (context-cropped) full sequence at each step
//                       with no KV cache.
// -----------------------------------------------------------------------------
torch::Tensor generateTextSimpleCached(
    gpt2mhl& model,
    torch::Tensor idx,
    int max_new_tokens,
    int context_size = 0,
    bool use_cache = true)
{
    model->eval();

    int64_t ctx_len = (context_size > 0)
        ? static_cast<int64_t>(context_size)
        : model->getContextLength();

    torch::NoGradGuard no_grad;

    if (use_cache) {
        model->reset_kv_cache();

        // Pre-fill with the full prompt (restricted to the last ctx_len tokens).
        auto start_col = std::max(static_cast<int64_t>(0), idx.size(1) - ctx_len);
        torch::Tensor input_tokens = idx.slice(/*dim=*/1, /*start=*/start_col);
        int64_t input_tokens_length = input_tokens.size(1);

        torch::Tensor logits = model->forward(input_tokens, /*use_cache=*/true);

        // Can't generate more than ctx_len total result tokens due to the
        // limitation of the (learnt) position embedding.
        int64_t max_generable = ctx_len - input_tokens_length;
        max_new_tokens = static_cast<int>(std::min<int64_t>(max_new_tokens, max_generable));

        for (int step = 0; step < max_new_tokens; ++step) {
            // Greedy decode: [batch, vocab] -> [batch, 1]
            torch::Tensor next_idx = logits.select(/*dim=*/1, /*index=*/-1)
                                         .argmax(/*dim=*/-1, /*keepdim=*/true);
            idx = torch::cat({idx, next_idx}, /*dim=*/1);

            // Feed only the single new token back; attention uses the KV-cache.
            logits = model->forward(next_idx, /*use_cache=*/true);
        }
    } else {
        for (int step = 0; step < max_new_tokens; ++step) {
            auto start_col = std::max(static_cast<int64_t>(0), idx.size(1) - ctx_len);
            torch::Tensor logits = model->forward(
                idx.slice(/*dim=*/1, /*start=*/start_col),
                /*use_cache=*/false
            );

            torch::Tensor next_idx = logits.select(/*dim=*/1, /*index=*/-1)
                                         .argmax(/*dim=*/-1, /*keepdim=*/true);
            idx = torch::cat({idx, next_idx}, /*dim=*/1);
        }
    }

    return idx;
}

void calculate_size(const gpt2mhl& model, const std::string& label)
{
    int64_t total_params = 0;
    for (const auto& p : model->parameters())
    {
        total_params += p.numel();
    }
    std::cout << label << ": Total number of parameters: " << total_params << std::endl;

    // Exclude output head parameters for weight tying
    auto out_head_params = model.get()->parameters();
    int64_t out_head_count = 0;
    for (const auto& p : out_head_params)
    {
        out_head_count += p.numel();
    }
    int64_t total_params_gpt2 = total_params - out_head_count;
    std::cout << "  Number of trainable parameters considering weight tying: "
              << total_params_gpt2 << std::endl;

    // Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    double total_size_bytes = static_cast<double>(total_params) * 4.0;
    double total_size_mb = total_size_bytes / (1024.0 * 1024.0);

    std::cout << "  Total size of the model: " << std::fixed << std::setprecision(2)
              << total_size_mb << " MB" << std::endl;
    std::cout << std::endl;
}

// -----------------------------------------------------------------------------
// MLA memory estimator (converted from the Python memory_estimator.py)
// Compares KV-cache memory for the Multi-Head Latent Attention (MLA) model
// against standard MHA and GQA.
// -----------------------------------------------------------------------------
namespace {

// Number of bytes per element for each supported dtype.
const std::map<std::string, int> DTYPE_BYTES_MLA = {
    {"fp32", 4},
    {"bf16", 2},
    {"fp16", 2},
    {"fp8",  1},
    {"int8", 1}
};

// Format bytes for display.  Use MB when the value is < 0.1 GB so that small
// KV caches (e.g. MLA's ~2.4 MB) aren't truncated to "0.00 GB" by rounding.
std::string convert_bytes_mla(long double n)
{
    long double gb = n / 1000.0L / 1000.0L / 1000.0L;
    std::ostringstream oss;
    if (gb < 0.1L)
    {
        long double mb = n / 1000.0L / 1000.0L;
        oss << std::fixed << std::setprecision(2) << mb << " MB";
    }
    else
    {
        oss << std::fixed << std::setprecision(2) << gb << " GB";
    }
    return oss.str();
}

// Standard KV-cache: per-head dim = ceil(emb_dim / n_heads), ×2 for K and V.
long double calc_kv_bytes_total_mla(
    int batch, int context_length, int emb_dim, int n_heads,
    int n_kv_heads, int n_layers, int bytes_per_elem)
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

// Simple MLA per-token compressed latent:
//   bytes ≈ batch × seqlen × n_layers × latent_dim × bytes_per_elem
long double calc_mla_bytes_total(
    int batch, int context_length, int n_layers, int latent_dim, int bytes_per_elem)
{
    return static_cast<long double>(batch) *
           static_cast<long double>(context_length) *
           static_cast<long double>(n_layers) *
           static_cast<long double>(latent_dim) *
           static_cast<long double>(bytes_per_elem);
}

} // end anonymous namespace

TEST_CASE("mlaMemoryEstimate_MHA_GQA_MLA")
{
    // ---- Default configuration (matches Python argparse defaults) ----
    int context_length = 1024;
    int emb_dim        = 768;      // required, example: gpt2
    int n_heads        = 12;       // required
    int n_layers       = 12;       // required
    int n_kv_groups    = 4;        // required
    int latent_dim     = 96;       // MLA per-token latent dimension
    int batch_size     = 1;
    std::string dtype  = "fp16";

    // ---- Input validation ----
    if (DTYPE_BYTES_MLA.find(dtype) == DTYPE_BYTES_MLA.end())
    {
        throw std::invalid_argument("Invalid dtype: " + dtype);
    }
    if (n_heads % n_kv_groups != 0)
    {
        throw std::invalid_argument("n_kv_groups must divide n_heads exactly.");
    }

    int bytes_per_elem = DTYPE_BYTES_MLA.at(dtype);
    int head_dim       = static_cast<int>(std::ceil(static_cast<double>(emb_dim) /
                                                    static_cast<double>(n_heads)));

    int n_kv_heads_mha = n_heads;
    int n_kv_heads_gqa = n_heads / n_kv_groups;

    long double total_mha = calc_kv_bytes_total_mla(
        batch_size, context_length, emb_dim, n_heads,
        n_kv_heads_mha, n_layers, bytes_per_elem);

    long double total_gqa = calc_kv_bytes_total_mla(
        batch_size, context_length, emb_dim, n_heads,
        n_kv_heads_gqa, n_layers, bytes_per_elem);

    long double total_mla = calc_mla_bytes_total(
        batch_size, context_length, n_layers, latent_dim, bytes_per_elem);

    long double ratio_mha_gqa = (total_gqa != 0.0L)
        ? (total_mha / total_gqa)
        : std::numeric_limits<long double>::infinity();
    long double savings_gqa   = (total_mha != 0.0L)
        ? (1.0L - (total_gqa / total_mha))
        : 0.0L;

    long double ratio_mha_mla = (total_mla != 0.0L)
        ? (total_mha / total_mla)
        : std::numeric_limits<long double>::infinity();
    long double savings_mla   = (total_mha != 0.0L)
        ? (1.0L - (total_mla / total_mha))
        : 0.0L;

    // ---- Output: Config ----
    std::cout << "\n==== Config ====\n";
    std::cout << std::left << std::setw(17) << "context_length" << ": " << context_length << "\n";
    std::cout << std::left << std::setw(17) << "emb_dim"        << ": " << emb_dim        << "\n";
    std::cout << std::left << std::setw(17) << "n_heads"        << ": " << n_heads        << "\n";
    std::cout << std::left << std::setw(17) << "n_layers"       << ": " << n_layers       << "\n";
    std::cout << std::left << std::setw(17) << "n_kv_groups"    << ": " << n_kv_groups    << "\n";
    std::cout << std::left << std::setw(17) << "latent_dim"     << ": " << latent_dim     << "\n";
    std::cout << std::left << std::setw(17) << "batch_size"     << ": " << batch_size     << "\n";
    std::cout << std::left << std::setw(17) << "dtype"          << ": " << dtype
              << " (" << bytes_per_elem << " Bytes/elem)\n";
    std::cout << std::left << std::setw(17) << "head_dim"       << ": " << head_dim       << "\n";
    std::cout << std::left << std::setw(17) << "GQA n_kv_heads" << ": " << n_kv_heads_gqa << "\n";
    std::cout << "\n";

    // ---- Output: KV-cache totals across all layers ----
    std::cout << "==== KV-cache totals across all layers ====\n";
    std::cout << "MHA total KV cache  : " << convert_bytes_mla(total_mha) << "\n";
    std::cout << "GQA total KV cache  : " << convert_bytes_mla(total_gqa) << "\n";
    std::cout << "MLA total KV cache  : " << convert_bytes_mla(total_mla) << "\n";
    std::cout << "Ratio (MHA / GQA)   : "
              << std::fixed << std::setprecision(2) << ratio_mha_gqa << "x\n";
    std::cout << "Savings (GQA vs MHA): "
              << std::fixed << std::setprecision(2) << (savings_gqa * 100.0L) << "%\n";
    std::cout << "Ratio (MHA / MLA)   : "
              << std::fixed << std::setprecision(2) << ratio_mha_mla << "x\n";
    std::cout << "Savings (MLA vs MHA): "
              << std::fixed << std::setprecision(2) << (savings_mla * 100.0L) << "%\n";
    std::cout << std::endl;
}

// -----------------------------------------------------------------------------
// Test case: "multiHeadLatentTextGeneration"
//
// Replaces the Python main() entry point.  Configures the GPT-2 style model
// with Multi-Head Latent Attention (MLA), encodes a starting prompt, generates
// tokens with the KV-cache, and reports timing / memory statistics.
// -----------------------------------------------------------------------------
TEST_CASE("multiHeadLatentTextGeneration")
{
    // ---- Tokenizer / data ----
    PreparedData data(std::string(DATASETS_DIR) + "gpt2.tiktoken");

    std::string start_context = "Hello, I am";
    torch::Tensor encoded_tensor = data.encodeBatch({start_context});
    int64_t prompt_len = encoded_tensor.size(1);
    std::cout << "\n" << std::string(50, '=') << "\n"
              << std::string(22, ' ') << "IN\n"
              << std::string(50, '=') << "\n";
    std::cout << "\nInput text: " << start_context << "\n";
    std::cout << "encoded_tensor.shape: [" << encoded_tensor.size(0)
              << ", " << encoded_tensor.size(1) << "]\n";

    // ---- Model configuration ----
    config cfg;
    int max_new_tokens = 200;
    cfg.vocab_size     = 50257;              // Vocabulary size
    cfg.context_length = max_new_tokens + static_cast<int>(prompt_len); // context window
    cfg.emb_dim        = 768;                // Embedding dimension
    cfg.n_heads        = 12;                 // Number of attention heads
    cfg.n_layer        = 12;                 // Number of layers
    cfg.drop_rate      = 0.0;                // Dropout rate
    cfg.qkv_bias       = false;              // Query-Key-Value bias
    cfg.latent_dim     = 0;                  // Default latent dim (auto: max(16, emb_dim/8))

    torch::manual_seed(123);
    gpt2mhl model(cfg);

    torch::Device device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA)
        : torch::Device(torch::kCPU);
    model->to(device, torch::kBFloat16);
    model->eval();

    encoded_tensor = encoded_tensor.to(device);

    // ---- Timing ----
    if (torch::cuda::is_available()) {
        torch::cuda::synchronize();
    }
    auto start = std::chrono::steady_clock::now();

    torch::Tensor token_ids = generateTextSimpleCached(
        model,
        encoded_tensor,
        /*max_new_tokens=*/200
    );

    if (torch::cuda::is_available()) {
        torch::cuda::synchronize();
    }
    auto end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end - start).count();

    // ---- Decode & report ----
    std::string decoded_text = data.decode(token_ids.cpu());

    std::cout << "\n\n" << std::string(50, '=') << "\n"
              << std::string(22, ' ') << "OUT\n"
              << std::string(50, '=') << "\n";
    std::cout << "\nOutput shape: [" << token_ids.size(0) << ", " << token_ids.size(1) << "]\n";
    std::cout << "Output length: " << token_ids.size(1) << "\n";
    std::cout << "Output text: " << decoded_text << "\n";

    std::cout << std::fixed << std::setprecision(2)
              << "\nTime: " << total_time << " sec\n";
    std::cout << std::setprecision(0)
              << static_cast<int>(token_ids.size(1) / total_time) << " tokens/sec\n";
    calculate_size(model, "gpt2-small (MLA)");
}