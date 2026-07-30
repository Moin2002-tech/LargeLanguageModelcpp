//
// Created by moinshaikh on 7/9/26.
//

#ifndef LARGELANGUAGEMODELCPP_TRANSFORMERBLOCKIMPL_HPP
#define LARGELANGUAGEMODELCPP_TRANSFORMERBLOCKIMPL_HPP

#include <torch/torch.h>
#include<AttentionMechanism/MultiHeadAttention.hpp>
#include"util.hpp"

class FeedForwardImpl : public torch::nn::Module
{
private:
    config cfg;
    torch::nn::Sequential layer{nullptr};
public:
    FeedForwardImpl(const config &cfg) : cfg(cfg)
    {
        layer= register_module("layer", torch::nn::Sequential(torch::nn::Linear(cfg.emb_dim,4 * cfg.emb_dim),
                                       torch::nn::GELU(),
                                       torch::nn::Linear(4 *cfg.emb_dim, cfg.emb_dim)));

    }
    torch::Tensor forward(torch::Tensor x)
    {
      return  layer->forward(x);
    }
};TORCH_MODULE(FeedForward);

class TransformerBlockImpl : public torch::nn::Module
{
private:
    config cfg;
    MultiHeadAttentionMechanism att{nullptr};
    FeedForward ff{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Dropout drop_shortcut{nullptr};

public:
    TransformerBlockImpl(const config &cfg);
    torch::Tensor forward(torch::Tensor x);
};TORCH_MODULE(TransformerBlock);

#endif //LARGELANGUAGEMODELCPP_TRANSFORMERBLOCKIMPL_HPP