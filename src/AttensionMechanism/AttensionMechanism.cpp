//
// Created by moinshaikh on 7/6/26.
//

#include <AttensionMechanism/AttensionMechanism.hpp>
#include <torch/torch.h>
#include <cmath>
#include<doctest.hpp>
SelfAttentionMechanismV2Impl::SelfAttentionMechanismV2Impl(uint d_in, uint d_out, bool qkv_bias)
    : d_in(d_in), d_out(d_out)
{
    W_query = register_module("W_query", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    W_key   = register_module("W_key",   torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    W_value = register_module("W_value", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
}

torch::Tensor SelfAttentionMechanismV2Impl::forward(torch::Tensor x)
{
    auto keys    = W_key->forward(x);
    auto queries = W_query->forward(x);
    auto values  = W_value->forward(x);

    // attn_scores = queries @ keys.T
    auto attn_scores = queries.matmul(keys.transpose(0, 1));

    // attn_weights = softmax(attn_scores / sqrt(d_out), dim=-1)
    auto attn_weights = torch::softmax(attn_scores / std::sqrt(static_cast<double>(d_out)), /*dim=*/1);

    // context_vec = attn_weights @ values
    auto context_vec = attn_weights.matmul(values);
    return context_vec;
}
TEST_CASE("SelfAttentionV2")
{
    torch::Tensor input = torch::tensor({{0.43, 0.15, 0.89},//your
                                          {0.55, 0.87, 0.66}, //journey
                                          {0.57, 0.85, 0.64}, //start
                                          {0.22, 0.58, 0.33},//with
                                          {0.77, 0.25, 0.10},//one
                                          {0.05, 0.80, 0.55}//step
      });
    uint d_in = input.size(1);
    uint d_out = 2;
    bool qkv_bias = false;
    torch::manual_seed(789);
    SelfAttentionMechanismV2 model(d_in, d_out, qkv_bias);
    auto attn_weights =  model->forward(input);

    std::cout<<"attn_weights= "<<attn_weights<<std::endl;
}

SelfAttentionMechanismV3Impl::SelfAttentionMechanismV3Impl(uint d_in, uint d_out, bool qkv_bias) : d_in(d_in), d_out(d_out), qkv_bias(qkv_bias)
{
    W_key = register_module("W_key",torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    W_query= register_module("W_query",torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    W_value = register_module("W_value",torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
}

torch::Tensor SelfAttentionMechanismV3Impl::forward(torch::Tensor x) {
    auto keys    = W_key->forward(x);
    auto queries = W_query->forward(x);
    auto values  = W_value->forward(x);
    auto attn_scores = queries.matmul(keys.transpose(0, 1));
    auto attn_weights = torch::softmax(attn_scores / (keys.size(-1)) *0.5,  -1);
    auto context_vec = attn_weights.matmul(values);
    return context_vec;

}

TEST_CASE("SelfAttentionV3") {
    torch::Tensor input = torch::tensor({{0.43, 0.15, 0.89},//your
                                          {0.55, 0.87, 0.66},//journey
                                          {0.57, 0.85, 0.64},//start
                                          {0.22, 0.58, 0.33},//with
                                          {0.77, 0.25, 0.10},//one
                                          {0.05, 0.80, 0.55}//step
      });

    uint d_in = input.size(1);
    uint d_out = 2;
    bool qkv_bias = false;
    torch::manual_seed(789);
    SelfAttentionMechanismV3 model(d_in, d_out, qkv_bias);
    auto attn_weights = model->forward(input);
    std::cout<<"attn_weights= "<<attn_weights<<std::endl;
}
