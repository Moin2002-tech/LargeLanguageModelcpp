//
// Created by moinshaikh on 8/10/26.
//
#include<Gpt2Untrained/Tests/MultiHeadLatent/include/TransformBlockV5.hpp>
#include<Gpt2Untrained/Tests/MultiHeadLatent/include/MultiHeadLatentV1.hpp>

TransformBlockV5Impl::TransformBlockV5Impl(config &cfg) : cfg(cfg) {
    att = register_module("att",MultiHeadLatentV1(cfg.emb_dim, cfg.emb_dim, cfg.drop_rate, cfg.n_heads,  cfg.qkv_bias,cfg.latent_dim));
    ff = register_module("ff",FeedForwardV3(cfg));
    layerNorm1 = register_module("layerNorm1",LayerNormV3(cfg.emb_dim));
    layerNorm2 = register_module("layerNorm2",LayerNormV3(cfg.emb_dim));
    dropoutShortcut = register_module("dropout_shortcut", torch::nn::Dropout(torch::nn::DropoutOptions(cfg.drop_rate)));
}


torch::Tensor TransformBlockV5Impl::forward(torch::Tensor x, bool use_cache)
{
    torch::Tensor shortcut =x;
    x = layerNorm1(x);


    x = att(x,use_cache);

    x= dropoutShortcut(x);
    x = x + shortcut;


    shortcut = x;
    x = layerNorm2(x);
    x = ff(x);
    x = dropoutShortcut(x);
    x =  x+ shortcut;


    return x;

}

void TransformBlockV5Impl::reset_cache()
{
    att->reset_cache();
}