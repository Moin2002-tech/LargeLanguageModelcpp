//
// Created by moinshaikh on 7/9/26.
//
#pragma once
#ifndef LARGELANGUAGEMODELCPP_DUMMYGPT2_HPP
#define LARGELANGUAGEMODELCPP_DUMMYGPT2_HPP

#include <torch/torch.h>
#include<iostream>
#include"util.hpp"
class Gpt2Impl : public torch::nn::Module {
private:
    config cfg;
    torch::nn::Embedding tokembed{nullptr}, postembed{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear outHead{nullptr};
    torch::nn::Sequential trfBlock{nullptr};
    torch::nn::LayerNorm layernorm{nullptr};
public:

    Gpt2Impl(config& cfg);
    torch::Tensor forward(torch::Tensor x);

};TORCH_MODULE(Gpt2);



#endif //LARGELANGUAGEMODELCPP_DUMMYGPT2_HPP
