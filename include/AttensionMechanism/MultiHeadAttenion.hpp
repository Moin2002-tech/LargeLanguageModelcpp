//
// Created by moinshaikh on 7/8/26.
//
#pragma once
#ifndef LARGELANGUAGEMODELCPP_MULTIHEADATTENION_HPP
#define LARGELANGUAGEMODELCPP_MULTIHEADATTENION_HPP

#include<torch/torch.h>
#include<torch/nn.h>
#include<cassert>

#include "CasualAttentionMechanism.hpp"
#include<GPT2LargeLanguageModel/util.hpp>



class MultiHeadAttentionMechanismImpl : public torch::nn::Module {
private:
    uint d_in;
    uint d_out;
    uint numHeads;
    uint headDim;
    int64_t context_length;
    double dropout;
    bool qkv_bias;
    torch::nn::Linear W_query{nullptr},W_value{nullptr},W_keys{nullptr};
    torch::nn::Linear out_proj{nullptr};
    torch::nn::Dropout dropout_layer{nullptr};
    torch::Tensor mask;
    config cfg;
public:
    MultiHeadAttentionMechanismImpl(uint d_in,
        uint d_out,
        int64_t context_length,
        uint numHeads,
        double dropout,
        bool qkv_bias = false);
    torch::Tensor forward(torch::Tensor x);
};TORCH_MODULE(MultiHeadAttentionMechanism);

#endif //LARGELANGUAGEMODELCPP_MULTIHEADATTENION_HPP