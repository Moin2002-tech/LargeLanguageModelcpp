//
// Created by moinshaikh on 8/5/26.
//

#include<Gpt2Untrained/Tests/GroupQueryAttention/include/TransformerBlockV4.hpp>


TransformerBlockV4Impl::TransformerBlockV4Impl(config &cfg)  :
d_in(cfg.emb_dim),
d_out(cfg.emb_dim),
numHeads(cfg.n_heads),
num_kv_groups(cfg.n_kv_groups),
dropout_rate(cfg.drop_rate),
qkv_bias(cfg.qkv_bias)
{
    att = register_module("att",GroupQueryAttentionV1(d_in, d_out, dropout_rate, numHeads, num_kv_groups, qkv_bias));
    ff = register_module("ff",FeedForwardV3(cfg));
    layerNorm1 = register_module("layerNorm1",LayerNormV3(cfg.emb_dim));
    layerNorm2 = register_module("layerNorm2",LayerNormV3(cfg.emb_dim));
    dropout_shortcut = register_module("dropout_shortcut", torch::nn::Dropout(torch::nn::DropoutOptions(cfg.drop_rate)));
}

torch::Tensor TransformerBlockV4Impl::forward(torch::Tensor x, bool use_Cache)
{
    auto shortcut = x;
    x = layerNorm1(shortcut);
    x = att->forward(x, use_Cache);
    x = dropout_shortcut(x);
    x =  x+ shortcut;

    shortcut = x;
    x =  layerNorm2(x);
    x = ff(x);
    x = dropout_shortcut(x);
    x =  x+ shortcut;

    return x;
    
}

void TransformerBlockV4Impl::reset_cache()
{
    att->reset_cache();
}
