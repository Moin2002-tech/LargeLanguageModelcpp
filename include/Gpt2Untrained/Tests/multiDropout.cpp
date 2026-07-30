//
// Created by moinshaikh on 7/22/26.
//

#include<torch/torch.h>
#include<torch/nn.h>

#include<GPT2LargeLanguageModel/util.hpp>
#include<GPT2LargeLanguageModel/TransformerBlock.hpp>
#include<GPT2LargeLanguageModel/Gpt2Model.hpp>
#include<AttentionMechanism/MultiHeadAttention.hpp>
#include<iostream>

#include<doctest.hpp>

/*
 * Configuration struct with separate dropout rates for embedding, attention,
 * and shortcut connections — matching the Python config pattern.
 */
struct ConfigV2 {
    int vocab_size = 50257;
    int context_length = 1024;
    int emb_dim = 768;
    int n_heads = 12;
    int n_layers = 12;
    float drop_rate_emb = 0.1;      // dropout for embedding layers
    float drop_rate_attn = 0.1;     // dropout for multi-head attention
    float drop_rate_shortcut = 0.1; // dropout for shortcut connections
    bool qkv_bias = false;
};


class TransformerBlockV2Impl : public torch::nn::Module
{
private:
    MultiHeadAttentionMechanism att{nullptr};
    FeedForward ff{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Dropout drop_shortcut{nullptr};

public:
    TransformerBlockV2Impl(const ConfigV2& cfg)
    {
        // Multi-head attention with its own dropout rate
        att = register_module("att", MultiHeadAttentionMechanism(
            static_cast<uint>(cfg.emb_dim),
            static_cast<uint>(cfg.emb_dim),
            cfg.context_length,
            static_cast<uint>(cfg.n_heads),
            cfg.drop_rate_attn,
            cfg.qkv_bias));

        // Feed-forward module (reuses existing FeedForward with base config)
        config base_cfg;
        base_cfg.emb_dim = cfg.emb_dim;
        ff = register_module("ff", FeedForward(base_cfg));

        // Layer norms
        norm1 = register_module("norm1", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({static_cast<int64_t>(cfg.emb_dim)})));
        norm2 = register_module("norm2", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({static_cast<int64_t>(cfg.emb_dim)})));

        // Shortcut dropout uses separate rate
        drop_shortcut = register_module("drop_shortcut",
            torch::nn::Dropout(torch::nn::DropoutOptions(cfg.drop_rate_shortcut)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        // Shortcut connection for attention block
        auto shortcut = x;
        x = norm1->forward(x);
        x = att->forward(x);
        x = drop_shortcut->forward(x);
        x = x + shortcut;

        // Shortcut connection for feed-forward block
        shortcut = x;
        x = norm2->forward(x);
        x = ff->forward(x);
        x = drop_shortcut->forward(x);
        x = x + shortcut;

        return x;
    }
};
TORCH_MODULE(TransformerBlockV2);


class GPTModelV2Impl : public torch::nn::Module
{
private:
    ConfigV2 cfg;
    torch::nn::Embedding tok_embed{nullptr}, pos_embed{nullptr};
    torch::nn::Dropout drop_emb{nullptr};
    torch::nn::Sequential trf_blocks{nullptr};
    torch::nn::LayerNorm final_norm{nullptr};
    torch::nn::Linear out_head{nullptr};

public:
    GPTModelV2Impl(const ConfigV2& cfg_) : cfg(cfg_)
    {
        // Token and positional embeddings
        tok_embed = register_module("tok_embed",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg.vocab_size, cfg.emb_dim)));
        pos_embed = register_module("pos_embed",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg.context_length, cfg.emb_dim)));

        // Embedding dropout (separate rate)
        drop_emb = register_module("drop_emb",
            torch::nn::Dropout(torch::nn::DropoutOptions(cfg.drop_rate_emb)));

        // Sequential of transformer blocks
        trf_blocks = register_module("trf_blocks", [this]() {
            torch::nn::Sequential seq;
            for (int i = 0; i < cfg.n_layers; ++i) {
                seq->push_back(TransformerBlockV2(cfg));
            }
            return seq;
        }());

        // Final layer norm
        final_norm = register_module("final_norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({static_cast<int64_t>(cfg.emb_dim)})));

        // Output head (no bias — matching Python: nn.Linear(emb_dim, vocab_size, bias=False))
        out_head = register_module("out_head",
            torch::nn::Linear(torch::nn::LinearOptions(cfg.emb_dim, cfg.vocab_size).bias(false)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto batch_size = x.size(0);
        auto seq_len = x.size(1);

        // Token + positional embeddings
        auto tok_embeds = tok_embed->forward(x);
        auto pos_embeds = pos_embed->forward(torch::arange(seq_len, x.device()));
        x = tok_embeds + pos_embeds;

        // Dropout on embeddings
        x = drop_emb->forward(x);

        // Through transformer blocks
        x = trf_blocks->forward(x);

        // Final norm and output projection
        x = final_norm->forward(x);
        x = out_head->forward(x);

        return x;
    }
};
TORCH_MODULE(GPTModelV2);

TEST_CASE("multidropout")
{
    /*
     * We defined a global drop_rate setting in the GPT_CONFIG_124M dictionary
     * to set the dropout rate in various places throughout the GPTModel architecture.
     * Change the code to specify a separate dropout value for the various dropout
     * layers throughout the model architecture.
     *
     * This test creates a GPTModelV2 with separate dropout rates for:
     *   - embedding layers       (drop_rate_emb)
     *   - multi-head attention    (drop_rate_attn)
     *   - shortcut connections    (drop_rate_shortcut)
     * and verifies the model runs a forward pass correctly.
     */

    // Configuration with separate dropout rates (matching Python: drop_rate_emb, drop_rate_attn, drop_rate_shortcut)
    ConfigV2 cfg;
    cfg.vocab_size = 50257;
    cfg.context_length = 1024;
    cfg.emb_dim = 768;
    cfg.n_heads = 12;
    cfg.n_layers = 12;
    cfg.drop_rate_emb = 0.1;
    cfg.drop_rate_attn = 0.2;      // different from embedding dropout
    cfg.drop_rate_shortcut = 0.15; // different from both
    cfg.qkv_bias = false;

    // Create model
    auto model = GPTModelV2(cfg);

    // Print model structure
    std::cout << model << std::endl;

    // Count total parameters
    int64_t total_params = 0;
    for (const auto& p : model->parameters()) {
        total_params += p.numel();
    }
    std::cout << "Total parameters: " << total_params << std::endl;

    // Create a dummy input batch: [1, 4] — 1 sample, 4 tokens
    auto input_ids = torch::randint(0, cfg.vocab_size, {1, 4});
    std::cout << "Input shape: " << input_ids.sizes() << std::endl;

    // Forward pass
    auto output = model->forward(input_ids);

    // Verify output shape: [batch_size, seq_len, vocab_size]
    REQUIRE(output.size(0) == 1);                          // batch_size
    REQUIRE(output.size(1) == 4);                          // seq_len
    REQUIRE(output.size(2) == cfg.vocab_size);             // vocab_size

    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "Test passed — model with separate dropout rates works correctly." << std::endl;
}