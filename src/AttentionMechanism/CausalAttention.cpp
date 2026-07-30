//
// Created by moinshaikh on 7/8/26.
//

#include<AttentionMechanism/CausalAttentionMechanism.hpp>
#include<limits>
#include<cmath>
#include<doctest.hpp>

CausalAttentionMechanismImpl::CausalAttentionMechanismImpl(uint d_in,
    uint d_out,
    int64_t context_length,
    double dropout,
    bool qkv_bias):
d_in(d_in),
d_out(d_out),
context_length(context_length),
qkv_bias(qkv_bias)
{
    W_key   = register_module("W_key",   torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    W_query = register_module("W_query", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    W_value = register_module("W_value", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    dropout_layer = register_module("dropout", torch::nn::Dropout(dropout));

    mask = register_buffer("mask",
        torch::triu(torch::ones({context_length, context_length}), 1));
}

torch::Tensor CausalAttentionMechanismImpl::forward(torch::Tensor x) {
    // x shape: (b, num_tokens, d_in)
    auto num_tokens = x.size(1);

    auto keys = W_key->forward(x);      // (b, num_tokens, d_out)
    auto queries = W_query->forward(x);  // (b, num_tokens, d_out)
    auto values = W_value->forward(x);   // (b, num_tokens, d_out)

    // attn_scores = queries @ keys.transpose(1, 2)
    auto attn_scores = queries.matmul(keys.transpose(1, 2));

    // mask out upper triangle
    auto mask_bool = mask.to(torch::kBool)
                         .slice(0, 0, num_tokens)
                         .slice(1, 0, num_tokens);
    attn_scores.masked_fill_(mask_bool, -std::numeric_limits<double>::infinity());

    // softmax with scaling
    auto attn_weights = torch::softmax(
        attn_scores / std::sqrt((double)keys.size(-1)), -1);

    // dropout
    attn_weights = dropout_layer->forward(attn_weights);

    // context_vec = attn_weights @ values
    auto context_vec = attn_weights.matmul(values);
    return context_vec;
}

TEST_CASE("CausalAttentionMechanism") {
    torch::Tensor input = torch::tensor({{0.43, 0.15, 0.89},//your
                                      {0.55, 0.87, 0.66}, //journey
                                      {0.57, 0.85, 0.64}, //start
                                      {0.22, 0.58, 0.33},//with
                                      {0.77, 0.25, 0.10},//one
                                      {0.05, 0.80, 0.55}//step
  });
    auto batch_size = torch::stack({input,input},0);
    torch::manual_seed(123);
    auto context_length =  batch_size.size(1);
    double dropout = 0.0;
    bool qkv_bias = false;
    CausalAttentionMechanism model(input.size(1),2,context_length,dropout,qkv_bias);
    auto context_vec = model->forward(batch_size);

    std::cout << context_vec << std::endl;
}