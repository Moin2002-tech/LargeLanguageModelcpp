//
// Created by moinshaikh on 8/3/26.
//
#include<torch/torch.h>

#include<Gpt2Untrained/Tests/GroupQueryAttention/include/GroupQueryAttentionV1.hpp>
#include<GPT2LargeLanguageModel/util.hpp>
#include<limits>
#include<cmath>


GroupQueryAttentionV1Impl::GroupQueryAttentionV1Impl(uint d_in, uint d_out, float dropout, int numHeads, int numKvGroups, bool qkv_bias)
    :   d_in(d_in),
        d_out(d_out),
        dropout_rate(dropout),
        numHeads(numHeads),
        numKvGroups(numKvGroups),
        qkv_bias(qkv_bias)

{
    head_dim = d_out / numHeads;
    groupSize = numHeads / numKvGroups;
    if (d_out % numHeads != 0)
    {
        throw std::invalid_argument("d_out must be divisible by numHeads");
    }
    if (numHeads % numKvGroups != 0) {
        throw std::invalid_argument("numHeads must be divisible by numKvGroups");
    }
    W_key = register_module("W_key", torch::nn::Linear(torch::nn::LinearOptions(d_in, numKvGroups * head_dim).bias(qkv_bias)));
    W_value = register_module("W_value", torch::nn::Linear(torch::nn::LinearOptions(d_in, numKvGroups * head_dim).bias(qkv_bias)));
    dropoutRate = register_module("dropout", torch::nn::Dropout(dropout));

    W_query = register_module("W_query", torch::nn::Linear(torch::nn::LinearOptions(d_in, numHeads * head_dim).bias(qkv_bias)));
    out_proj = register_module("out_proj", torch::nn::Linear(torch::nn::LinearOptions(numHeads * head_dim, d_out).bias(qkv_bias)));

    cache_k = register_buffer("cache_k", torch::Tensor());
    cache_v = register_buffer("cache_v", torch::Tensor());
}


torch::Tensor GroupQueryAttentionV1Impl::forward(torch::Tensor x, bool use_Cache)
{
    auto num_tokens = x.size(1);
    auto b = x.size(0);

    auto queries = W_query(x);
    auto keys = W_key(x);
    auto values = W_value(x);


    queries = queries.view({b, num_tokens,numHeads,head_dim}).transpose(1,2);
    auto keys_new = keys.view({b, num_tokens,numKvGroups,head_dim}).transpose(1,2);
    auto values_value = values.view({b, num_tokens,numKvGroups,head_dim}).transpose(1,2);

    // keys_base, values_base used for attention after caching
    torch::Tensor keys_base, values_base;

    if (use_Cache) {

        if (!cache_k.defined()) {

            cache_k = keys_new;
            cache_v = values_value;
        } else {

            cache_k = torch::cat({cache_k, keys_new}, 2);
            cache_v = torch::cat({cache_v, values_value}, 2);
        }

        keys_base = cache_k;
        values_base = cache_v;
    } else
    {
        // keys_base, values_base = keys_new, values_new
        keys_base = keys_new;
        values_base = values_value;


        if (cache_k.defined() || cache_v.defined()) {
            // self.cache_k, self.cache_v = None, None
            cache_k = torch::Tensor();
            cache_v = torch::Tensor();
            // self.ptr_current_pos = 0
            current_ptr = 0;
        }
    }
    // Expand keys and values to match the number of heads
    // Shape: (b, num_heads, num_tokens, head_dim)
    keys = keys_base.repeat_interleave(groupSize,1);
    values = values_base.repeat_interleave(groupSize,1);


    //For example, before repeat_interleave along dim=1 (query groups):
    //  [K1, K2]
    //After repeat_interleave (each query group is repeated group_size times):
    //  [K1, K1, K2, K2]
    //If we used regular repeat instead of repeat_interleave, we'd get:
    //  [K1, K2, K1, K2]

        //Compute scaled dot-product attention (aka self-attention) with a causal mask
        //Shape: (b, num_heads, num_tokens, num_tokens)
    auto attn_score = queries.matmul(keys.transpose(2,3));

    auto numTokens_Q = queries.size(-2);
    auto numTokens_K = keys.size(-2);
    torch::Device device = queries.device();
    torch::Tensor q_position;
    if (use_Cache)
    {
        q_position = torch::arange(current_ptr,current_ptr + numTokens_Q,torch::TensorOptions(device).dtype(torch::kLong));
        current_ptr += numTokens_Q;
    }
    else
    {
        q_position = torch::arange(numTokens_Q,torch::TensorOptions(device).dtype(torch::kLong));
        current_ptr = 0;
    }
    auto k_position = torch::arange(numTokens_K, torch::TensorOptions(device).dtype(torch::kLong));
    auto mask = q_position.unsqueeze(-1) < k_position.unsqueeze(0);
    attn_score = attn_score.masked_fill(mask,-std::numeric_limits<float>::infinity());


    // Scale attention scores by sqrt(head_dim) and apply softmax
    double scale = std::sqrt(keys.size(-1));  // head_dim
    auto attn_weights = torch::softmax(attn_score / scale, /*dim=*/-1);
    TORCH_CHECK(keys.size(-1) == head_dim, "keys last dim must equal head_dim");
    attn_weights = dropoutRate(attn_weights);

    // Shape: (b, num_tokens, num_heads, head_dim)
    auto context_vec = (attn_weights.matmul(values)).transpose(1, 2);

    // Combine heads, where d_out = num_heads * head_dim
    context_vec = context_vec.contiguous().view({b, num_tokens, d_out});
    context_vec = out_proj(context_vec);  // optional projection

    return context_vec;

}

void GroupQueryAttentionV1Impl::reset_cache()
{
    cache_k = torch::Tensor();
    cache_v = torch::Tensor();
    current_ptr = 0;
}
