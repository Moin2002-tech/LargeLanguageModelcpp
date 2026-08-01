//
// Created by moinshaikh on 7/31/26.
//
#include<Gpt2Untrained/Tests/kv_cache/include/LayerNormV2.hpp>


LayerNormV2Impl::LayerNormV2Impl(config &cfg) : cfg(cfg)
{
    scale =     register_parameter("scale", torch::ones({cfg.emb_dim}));
    shift =     register_parameter("shift",torch::zeros({cfg.emb_dim}));
}

torch::Tensor LayerNormV2Impl::forward(torch::Tensor x)
{
    auto mean = x.mean(-1,true);
    auto variance = x.var(-1,false,true);
    auto norm_x = (x-mean) / torch::sqrt(variance + eps);
    return (scale * norm_x + shift);
}
