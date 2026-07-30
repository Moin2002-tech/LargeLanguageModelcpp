//
// Created by moinshaikh on 7/9/26.
//
#include<GPT2LargeLanguageModel/TransformerBlock.hpp>


TransformerBlockImpl::TransformerBlockImpl(const config &cfg) : cfg(cfg)
{
    att = register_module("att", MultiHeadAttentionMechanism(
            cfg.emb_dim, cfg.emb_dim, cfg.context_length, cfg.n_heads, cfg.drop_rate, cfg.qkv_bias));

    ff = register_module("ff", FeedForward(cfg));

    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.emb_dim})));
    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.emb_dim})));

    drop_shortcut = register_module("drop_shortcut", torch::nn::Dropout(torch::nn::DropoutOptions(cfg.drop_rate)));
}

torch::Tensor TransformerBlockImpl::forward(torch::Tensor x)
{
    // Shortcut connection for attention block
    torch::Tensor shortcut = x;
    x = norm1->forward(x);
    x = att->forward(x);
    x = drop_shortcut->forward(x);
    x = x + shortcut;  // Add the original input back

    // Shortcut connection for feed forward block
    shortcut = x;
    x = norm2->forward(x);
    x = ff->forward(x);
    x = drop_shortcut->forward(x);
    x = x + shortcut;  // Add the original input back

    return x;
}