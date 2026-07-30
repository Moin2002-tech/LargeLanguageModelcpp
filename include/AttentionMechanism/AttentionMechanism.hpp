//
// Created by moinshaikh on 7/6/26.
//

#ifndef LARGELANGUAGEMODELCPP_ATTENTIONMECHANISM_HPP
#define LARGELANGUAGEMODELCPP_ATTENTIONMECHANISM_HPP

#include<iostream>
#include<torch/torch.h>
#include<tiktoken.hpp>

class SelfAttentionMechanismV2Impl : public torch::nn::Module {
private:
    uint d_in, d_out;
    torch::nn::Linear W_query{nullptr}, W_key{nullptr}, W_value{nullptr};

public:
    SelfAttentionMechanismV2Impl(uint d_in, uint d_out, bool qkv_bias=false);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SelfAttentionMechanismV2);

class SelfAttentionMechanismV3Impl : public torch::nn::Module {
private:
    uint d_in, d_out;
    bool qkv_bias;
    torch::nn::Linear W_query{nullptr}, W_key{nullptr}, W_value{nullptr};
public:
    SelfAttentionMechanismV3Impl(uint d_in, uint d_out, bool qkv_bias);
    torch::Tensor forward(torch::Tensor x);
}; TORCH_MODULE(SelfAttentionMechanismV3);

#endif //LARGELANGUAGEMODELCPP_ATTENTIONMECHANISM_HPP