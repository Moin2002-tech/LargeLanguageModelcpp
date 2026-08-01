//
// Created by moinshaikh on 7/30/26.
//


#include <Gpt2Untrained/Tests/kv_cache/include/MultiHeadAttentionV2.hpp>



MultiHeadAttentionV2Impl::MultiHeadAttentionV2Impl(uint d_in,
    uint d_out,
    int64_t context_length,
    double dropout,
    uint numHeads,
    bool qkv_bias,
    int64_t max_seq_length,
    int64_t window_size)
:
d_in(d_in),
d_out(d_out),
numHeads(numHeads),
context_length(context_length),
dropout(dropout),
qkv_bias(qkv_bias),
max_seq_length(max_seq_length > 0 ? max_seq_length : context_length),
window_size(window_size > 0 ? window_size : this->max_seq_length)
{
    headDim =d_out / numHeads;
    if (d_out % numHeads != 0)
    {
        throw std::invalid_argument("d_out must be divisible by numHeads");
    }
    W_query = register_module("W_query", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    W_value =  register_module("W_value", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    W_keys =  register_module("W_keys", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    out_proj = register_module("out_proj", torch::nn::Linear(torch::nn::LinearOptions(d_out, d_out)));
    dropout_layer =  register_module("dropout", torch::nn::Dropout(dropout));
    mask = register_buffer("mask",torch::triu(torch::ones({context_length,context_length}),1));

    //kv cache — initialized as undefined tensors
    cache_k = register_buffer("cache_k", torch::Tensor());
    cache_v = register_buffer("cache_v", torch::Tensor());
}

torch::Tensor MultiHeadAttentionV2Impl::forward(torch::Tensor x, bool use_cache)
{
    auto b = x.size(0);
    auto num_tokens = x.size(1);

    // 1. Linear projections — separate _new variables for fresh projections
    auto keys_new = W_keys->forward(x);    // (b, num_tokens, d_out)
    auto values_new = W_value->forward(x); // (b, num_tokens, d_out)
    auto queries = W_query->forward(x);    // (b, num_tokens, d_out)

    // 2. Reshape: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
    keys_new = keys_new.view({b, num_tokens, numHeads, headDim});
    values_new = values_new.view({b, num_tokens, numHeads, headDim});
    queries = queries.view({b, num_tokens, numHeads, headDim});

    // 3. Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
    keys_new = keys_new.transpose(1, 2);
    values_new = values_new.transpose(1, 2);
    queries = queries.transpose(1, 2);

    // 4. Declare the final keys/values that will be used for attention
    torch::Tensor keys, values;

    // 5. KV-cache logic
    if (use_cache) {
        // Prevent ptr_cur from becoming negative
        if (num_tokens > window_size) {
            throw std::invalid_argument(
                "Input chunk size (" + std::to_string(num_tokens) +
                ") exceeds KV cache window size (" + std::to_string(window_size) + ")."
            );
        }

        // Initialize cache if undefined or wrong batch size
        if (!cache_k.defined() || cache_k.size(0) != b) {
            cache_k = torch::zeros({b, numHeads, window_size, headDim}, x.device());
            cache_v = torch::zeros_like(cache_k);
            current_ptr = 0;
        }

        // Overflow: discard oldest tokens if incoming chunk won't fit
        if (current_ptr + num_tokens > window_size) {
            int64_t overflow = current_ptr + num_tokens - window_size;
            cache_k.index_put_(
                {torch::indexing::Slice(), torch::indexing::Slice(),
                 torch::indexing::Slice(0, window_size - overflow), torch::indexing::Slice()},
                cache_k.index({"...", torch::indexing::Slice(overflow, torch::indexing::None), "..."}).clone()
            );
            cache_v.index_put_(
                {torch::indexing::Slice(), torch::indexing::Slice(),
                 torch::indexing::Slice(0, window_size - overflow), torch::indexing::Slice()},
                cache_v.index({"...", torch::indexing::Slice(overflow, torch::indexing::None), "..."}).clone()
            );
            current_ptr -= overflow;
        }

        // Write new keys/values into the cache
        cache_k.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(),
             torch::indexing::Slice(current_ptr, current_ptr + num_tokens), torch::indexing::Slice()},
            keys_new
        );
        cache_v.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(),
             torch::indexing::Slice(current_ptr, current_ptr + num_tokens), torch::indexing::Slice()},
            values_new
        );
        current_ptr += num_tokens;

        // Read effective keys/values from cache (only the valid portion)
        keys = cache_k.index(
            {torch::indexing::Slice(), torch::indexing::Slice(),
             torch::indexing::Slice(0, current_ptr), torch::indexing::Slice()}
        );
        values = cache_v.index(
            {torch::indexing::Slice(), torch::indexing::Slice(),
             torch::indexing::Slice(0, current_ptr), torch::indexing::Slice()}
        );
    } else {
        // No cache: use fresh projections directly
        keys = keys_new;
        values = values_new;
        current_ptr = 0;
    }

    // 6. Compute scaled dot-product attention
    auto attn_scores = queries.matmul(keys.transpose(2, 3)); // (b, num_heads, num_tokens, K)

    // 7. Causal mask with cache offset
    int64_t K = attn_scores.size(-1);
    torch::Tensor causal_mask;
    if (num_tokens == K) {
        // No cache: standard triangular mask
        causal_mask = torch::triu(torch::ones({num_tokens, K},
            torch::dtype(torch::kBool).device(x.device())), 1);
    }
    else
    {
        // Cache: offset the diagonal by (K - num_tokens)
        int64_t offset = K - num_tokens;
        auto row_idx = torch::arange(num_tokens, x.device()).unsqueeze(1); // (num_tokens, 1)
        auto col_idx = torch::arange(K, x.device()).unsqueeze(0);          // (1, K)
        causal_mask = (row_idx + offset) < col_idx; // True where j > i+offset
    }

    // Apply mask
    attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -std::numeric_limits<float>::infinity());

    // 8. Softmax + scaling + dropout
    auto attn_weights = torch::softmax(attn_scores / std::sqrt(static_cast<double>(headDim)), -1);
    attn_weights = dropout_layer->forward(attn_weights);

    // 9. Weighted sum of values: (b, num_heads, num_tokens, head_dim)
    auto context_vec = attn_weights.matmul(values).transpose(1, 2);
    // Now shape: (b, num_tokens, num_heads, head_dim)

    // 10. Combine heads and project output
    context_vec = context_vec.contiguous().view({b, num_tokens, d_out});
    context_vec = out_proj->forward(context_vec);

    return context_vec;
}

void MultiHeadAttentionV2Impl::reset_cache()
{
    cache_k = torch::Tensor(); // set to undefined (equivalent to Python's None)
    cache_v = torch::Tensor();
    current_ptr = 0;
}
