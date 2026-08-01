//
// Created by moinshaikh on 7/30/26.
//

#ifndef LARGELANGUAGEMODELCPP_LAYERNORMV2_HPP
#define LARGELANGUAGEMODELCPP_LAYERNORMV2_HPP

#include<torch/torch.h>
#include<torch/nn.h>
#include<GPT2LargeLanguageModel/util.hpp>

class LayerNormV2Impl : public torch::nn::Module
{
private:
    torch::Tensor scale,shift;
    config cfg;
    float eps = 1e-5;
public:
    LayerNormV2Impl(config& cfg);
    torch::Tensor forward(torch::Tensor x);
};TORCH_MODULE(LayerNormV2);

#endif //LARGELANGUAGEMODELCPP_LAYERNORMV2_HPP
