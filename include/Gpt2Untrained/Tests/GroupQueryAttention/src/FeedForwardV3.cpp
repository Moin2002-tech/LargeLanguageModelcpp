//
// Created by moinshaikh on 8/5/26.
//
#include<Gpt2Untrained/Tests/GroupQueryAttention/include/FeedForwardV3.hpp>
#include<Gpt2Untrained/Tests/GroupQueryAttention/include/GELUV3.hpp>


FeedForwardV3Impl::FeedForwardV3Impl(config &cfg) : cfg(cfg) {
    layers = register_module("layers",torch::nn::Sequential(
        torch::nn::Linear(torch::nn::LinearOptions(cfg.emb_dim,4 * cfg.emb_dim)),
        GELUV3(),
        torch::nn::Linear(torch::nn::LinearOptions(4* cfg.emb_dim,cfg.emb_dim))
        ));
}

torch::Tensor FeedForwardV3Impl::forward(torch::Tensor x) {
    return layers->forward(x);
}