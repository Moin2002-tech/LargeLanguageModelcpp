//
// Created by moinshaikh on 7/31/26.
//

#ifndef LARGELANGUAGEMODELCPP_TRANSFORMBLOCKV2_HPP
#define LARGELANGUAGEMODELCPP_TRANSFORMBLOCKV2_HPP


#include"MultiHeadAttentionV2.hpp"
#include"LayerNormV2.hpp"
#include"FeedForward.hpp"

#include<torch/torch.h>
#include<torch/nn.h>

class TransformBlockV2Impl : public torch::nn::Module {
private:
    config cfg;
    MultiHeadAttentionV2 att{ nullptr};
    FeedForwardV2 ff{ nullptr };
    LayerNormV2 layerNorm1{ nullptr };
    LayerNormV2 layerNorm2{ nullptr };
    torch::nn::Dropout dropout_shotcut{nullptr};

public:

    TransformBlockV2Impl(config &cfg);
    torch::Tensor forward(torch::Tensor x, bool use_cache = false);
    void reset_cache();   // wraps att->reset_cache() from outside


};TORCH_MODULE(TransformBlockV2);
#endif //LARGELANGUAGEMODELCPP_TRANSFORMBLOCKV2_HPP
