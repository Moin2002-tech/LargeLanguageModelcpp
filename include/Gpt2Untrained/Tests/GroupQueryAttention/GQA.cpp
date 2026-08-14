//
// Created by moinshaikh on 8/5/26.
//
#include<Gpt2Untrained/Tests/GroupQueryAttention/include/Gpt2ModelV3.hpp>
#include<Gpt2Untrained/dataPreparation.hpp>
#include "doctest.hpp"

torch::Tensor generateTextSimpleCached(
    Gpt2ModelV3& model,
    torch::Tensor idx,
    int max_new_tokens,
    int context_size = 0,
    bool use_cache = true
)
{
    model->eval();

    int64_t ctx_len = (context_size > 0)
        ? static_cast<int64_t>(context_size)
        : model->getContextLength();
    int64_t kv_window_size = model->getKvWindowSize();

    torch::NoGradGuard no_grad;

    if (use_cache) {
        model->reset_kv_cache();

        // Restrict the input to the last `ctx_len` tokens
        auto start_col = std::max(static_cast<int64_t>(0), idx.size(1) - ctx_len);
        torch::Tensor input_tokens = idx.slice(/*dim=*/1, /*start=*/start_col);
        int64_t input_tokens_length = input_tokens.size(1);

        // Prefill: process large prompts in chunks of `kv_window_size`
        torch::Tensor logits;
        for (int64_t i = 0; i < input_tokens_length; i += kv_window_size) {
            torch::Tensor chunk = input_tokens.slice(/*dim=*/1, /*start=*/i, /*end=*/i + kv_window_size);
            logits = model->forward(chunk, /*use_cache=*/true);
        }

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


void calculate_size(const Gpt2ModelV3& model, const std::string& label)
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

TEST_CASE("groupQueryAttention")
{
PreparedData data(MODELS_DIR "/gpt2.tiktoken");


    std::string start_context = "Hello, I am";
    torch::Tensor encoded_tensor = data.encodeBatch({start_context});
    std::cout << "\nInput text: " << start_context << "\n";
    std::cout << "Encoded input text: " << encoded_tensor << "\n";

    config cfg;
    int max_new_tokens = 200;
    cfg.vocab_size    = 50257;   // Vocabulary size
    cfg.context_length = 1024 ;   // Context length
    cfg.emb_dim       = 768;     // Embedding dimension
    cfg.n_heads       = 12;      // Number of attention heads
    cfg.n_layer       = 12;      // Number of layers
    cfg.drop_rate     = 0.1;     // Dropout rate
    cfg.qkv_bias      = false;   // Query-Key-Value bias
    cfg.kv_window_size = 1024;   // KV cache window size

    Gpt2ModelV3 model(cfg);

    auto start = std::chrono::steady_clock::now();

    torch::Tensor token_ids = generateTextSimpleCached(
        model,
        encoded_tensor,
        /*max_new_tokens=*/200
    );

    auto end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end - start).count();

    // Decode back to text (tokenizer.decode(token_ids.squeeze(0).tolist()))
    std::string decoded_text = data.decode(token_ids);

    std::cout << "\n\nOutput: " << token_ids << "\n";
    std::cout << "Output length: " << token_ids.size(1) << "\n";
    std::cout << "Output text: " << decoded_text << "\n";

    std::cout << "\nTime: " << total_time << " sec\n";
    std::cout << static_cast<int>(token_ids.size(1) / total_time) << " tokens/sec\n";
    calculate_size(model,"gpt2-small");


}
