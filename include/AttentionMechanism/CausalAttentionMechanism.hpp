//
// Created by moinshaikh on 7/8/26.
//
#pragma once
#ifndef LARGELANGUAGEMODELCPP_CAUSALATTENTIONMECHANISM_HPP
#define LARGELANGUAGEMODELCPP_CAUSALATTENTIONMECHANISM_HPP

#include<torch/torch.h>
#include<cmath>

class CausalAttentionMechanismImpl : public torch::nn::Module {
private:
  uint d_in, d_out;
  torch::nn::Linear W_key{nullptr}, W_value{nullptr}, W_query{nullptr};
  torch::nn::Dropout dropout_layer{nullptr};
  torch::Tensor mask;
  int64_t context_length;
  bool qkv_bias;
public:
  CausalAttentionMechanismImpl(uint d_in, uint d_out, int64_t context_length, double dropout, bool qkv_bias = false);

  torch::Tensor forward(torch::Tensor x);

};TORCH_MODULE(CausalAttentionMechanism);

#endif //LARGELANGUAGEMODELCPP_CAUSALATTENTIONMECHANISM_HPP