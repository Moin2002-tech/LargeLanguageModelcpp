//
// Created by moinshaikh on 7/31/26.
//
#include<Gpt2Untrained/Tests/kv_cache/include/FeedForward.hpp>


FeedForwardV2Impl::FeedForwardV2Impl(config &cfg) :  cfg(cfg) {
    layers = register_module("layers", torch::nn::Sequential(
        torch::nn::Linear(cfg.emb_dim,4 * cfg.emb_dim),
        GELU(),
        torch::nn::Linear(4 *cfg.emb_dim, cfg.emb_dim))
        );
}

torch::Tensor FeedForwardV2Impl::forward(torch::Tensor x) 
{
    return layers->forward(x);
}