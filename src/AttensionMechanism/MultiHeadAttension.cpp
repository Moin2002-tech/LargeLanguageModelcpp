//
// Created by moinshaikh on 7/8/26.
//


#include<AttensionMechanism/MultiHeadAttenion.hpp>
#include<doctest.hpp>
#include<stdexcept>
#include<limits>
#include<cmath>


MultiHeadAttentionMechanismImpl::MultiHeadAttentionMechanismImpl(uint d_in, uint d_out,
        int64_t context_length,
        uint numHeads,
        double dropout,
        bool qkv_bias) :
        d_in(d_in),
        d_out(d_out),
        context_length(context_length),
        numHeads(numHeads),
        headDim(d_out / numHeads),
        qkv_bias(qkv_bias)
{
        if (d_out % numHeads != 0)
        {
           throw std::invalid_argument("d_out must be divisible by numHeads");
        }
        W_query = torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias));
        W_value =  torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias));
        W_keys =  torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias));
        out_proj = torch::nn::Linear(torch::nn::LinearOptions(d_out, d_out));
        dropout_layer =  torch::nn::Dropout(dropout);
        mask = register_buffer("mask",torch::triu(torch::ones({context_length,context_length}),1));
}

torch::Tensor MultiHeadAttentionMechanismImpl::forward(torch::Tensor x)
{

    auto b = x.size(0);
    auto num_tokens = x.size(1);

    // Linear projections
    auto keys = W_keys->forward(x);
    auto queries = W_query->forward(x);
    auto values = W_value->forward(x);

    // Split last dim into

    keys = keys.view({b, num_tokens, (int64_t)numHeads, (int64_t)headDim});
    queries = queries.view({b, num_tokens, (int64_t)numHeads, (int64_t)headDim});
    values = values.view({b, num_tokens, (int64_t)numHeads, (int64_t)headDim});

    // Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
    keys = keys.transpose(1, 2);
    queries = queries.transpose(1, 2);
    values = values.transpose(1, 2);

    // attn_scores = queries @ keys.transpose(2, 3)  -> (b, num_heads, num_tokens, num_tokens)
    auto attn_scores = queries.matmul(keys.transpose(2, 3));

    // Causal mask: slice to current num_tokens and convert to bool
    auto mask_bool = mask.to(torch::kBool)
                         .slice(0, 0, num_tokens)
                         .slice(1, 0, num_tokens);
    attn_scores.masked_fill_(mask_bool, -std::numeric_limits<double>::infinity());

    // Softmax with scaling
    auto attn_weights = torch::softmax(
        attn_scores / std::sqrt((double)keys.size(-1)), -1);

    // Dropout
    attn_weights = dropout_layer->forward(attn_weights);

    // context_vec: (b, num_heads, num_tokens, head_dim)
    auto context_vec = attn_weights.matmul(values);

    // Transpose back: (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
    context_vec = context_vec.transpose(1, 2);

    // Combine heads: (b, num_tokens, num_heads * head_dim) = (b, num_tokens, d_out)
    context_vec = context_vec.contiguous().view({b, num_tokens, (int64_t)d_out});

    // Optional output projection
    context_vec = out_proj->forward(context_vec);

    return context_vec;
}

TEST_CASE("MultiHeadAttentionMechanism") {
    torch::Tensor input = torch::tensor({{0.43, 0.15, 0.89},//your
                                        {0.55, 0.87, 0.66}, //journey
                                        {0.57, 0.85, 0.64}, //start
                                        {0.22, 0.58, 0.33},//with
                                        {0.77, 0.25, 0.10},//one
                                        {0.05, 0.80, 0.55}//step
    });

   auto batch = torch::stack({input,input}, 0);
    auto context_length =  batch.size(1);
    double dropout = 0.0;
   uint d_in = input.size(1);
    uint d_out = 2;
    torch::manual_seed(123);
    MultiHeadAttentionMechanism model(d_in, d_out,  context_length,2, dropout ,false);
    auto conVec = model->forward(batch);
    std::cout << conVec << std::endl;
}