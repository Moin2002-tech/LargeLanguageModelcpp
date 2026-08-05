//
// Created by moinshaikh on 8/5/26.
//


#include<Gpt2Untrained/Tests/GroupQueryAttention/include/LayerNormV3.hpp>


LayerNormV3Impl::LayerNormV3Impl(int emb_dim) : emb_dim(emb_dim) {
    scale = register_parameter("scale",torch::ones(emb_dim));
    shift = register_parameter("shift",torch::zeros(emb_dim));
}

torch::Tensor LayerNormV3Impl::forward(torch::Tensor x)
{
    auto mean = x.mean(-1,true);
    auto variance = x.var(-1,false,true);
    auto norm_x = (x -mean) /torch::sqrt(variance + eps);

    return (scale * norm_x) + shift;
}
