//
// Created by moinshaikh on 7/31/26.
//


#include<Gpt2Untrained/Tests/kv_cache/include/TransformBlockV2.hpp>


TransformBlockV2Impl::TransformBlockV2Impl(config &cfg) :cfg(cfg)
{
    att = register_module("att", MultiHeadAttentionV2(
        cfg.emb_dim,
        cfg.emb_dim,
        cfg.context_length,
        cfg.drop_rate,
        cfg.n_heads,
        cfg.qkv_bias,
        cfg.context_length,
        cfg.kv_window_size
        ));

    ff = register_module("ff", FeedForwardV2(cfg));

    layerNorm1 = register_module("layerNorm1", LayerNormV2(cfg));
    layerNorm2 = register_module("layerNorm2", LayerNormV2(cfg));

    dropout_shotcut = register_module("dropout_shotcut", torch::nn::Dropout(torch::nn::DropoutOptions(cfg.drop_rate)));
}

torch::Tensor TransformBlockV2Impl::forward(torch::Tensor x, bool use_cache) {
    // Shortcut connection for attention block
    torch::Tensor shortcut = x;
    x = layerNorm1->forward(x);
    x = att->forward(x, use_cache);
    x = dropout_shotcut->forward(x);
    x = x + shortcut;  // Add the original input back

    // Shortcut connection for feed forward block
    shortcut = x;
    x = layerNorm2->forward(x);
    x = ff->forward(x);
    x = dropout_shotcut->forward(x);
    x = x + shortcut;  // Add the original input back

    return x;
}

void TransformBlockV2Impl::reset_cache()
{
    att->reset_cache();
}