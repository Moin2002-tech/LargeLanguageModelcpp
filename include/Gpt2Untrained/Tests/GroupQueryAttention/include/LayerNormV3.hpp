//
// Created by moinshaikh on 8/5/26.
//

#ifndef LARGELANGUAGEMODELCPP_LAYERNORMV3_HPP
#define LARGELANGUAGEMODELCPP_LAYERNORMV3_HPP
#include "torch/nn/module.h"

class LayerNormV3Impl : public torch::nn::Module
{
private:
    int emb_dim;
    float eps = static_cast<float>(1e-5);
    torch::Tensor scale,shift;
public:
    explicit LayerNormV3Impl(int emb_dim);
    torch::Tensor forward(torch::Tensor x);

};TORCH_MODULE(LayerNormV3);

#endif //LARGELANGUAGEMODELCPP_LAYERNORMV3_HPP
