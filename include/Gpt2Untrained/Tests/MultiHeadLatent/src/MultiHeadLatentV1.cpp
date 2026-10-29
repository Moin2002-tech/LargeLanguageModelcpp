//
// Created by moinshaikh on 8/8/26.
//


#include<Gpt2Untrained/Tests/MultiHeadLatent/include/MultiHeadLatentV1.hpp>
#include <limits>
#include <cmath>
#include<TrainedGpt/textGeneration/LoadUtil.hpp>

MultiHeadLatentV1Impl::MultiHeadLatentV1Impl(int d_in,
    int d_out,
    float dropRate,
    int numHeads,
    bool qkv_bias,
    int latentDims) : d_in(d_in),
d_out(d_out),
dropRate(dropRate),
numHeads(numHeads),
qkv_bias(qkv_bias),
latentDims(latentDims)
{
    headDims = d_out / numHeads;
    if (d_out % numHeads != 0) {
        throw std::invalid_argument("d_out must be divisible by numHeads");
    }
    if (this->latentDims == 0) {
      this->latentDims = std::max(16, d_out / 8);
    }
    W_query = register_module("W_query",torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)));
    W_DKV = register_module("W_DKV", torch::nn::Linear(torch::nn::LinearOptions(d_in,this->latentDims).bias(qkv_bias)));
    W_UK = register_module("W_UK", torch::nn::Linear(torch::nn::LinearOptions(this->latentDims,d_out).bias(qkv_bias)));
    W_UV = register_module("W_UV", torch::nn::Linear(torch::nn::LinearOptions(this->latentDims,d_out).bias(qkv_bias)));

    outProj = register_module("outProj",torch::nn::Linear(torch::nn::LinearOptions(d_out,d_out)));
    dropout = register_module("dropout",torch::nn::Dropout(dropRate));

    cache_c_kv = register_buffer("cache_c_kv",torch::Tensor());

}

void MultiHeadLatentV1Impl::reset_cache() {
    cache_c_kv = torch::Tensor();
    currentPos = 0;
}

torch::Tensor MultiHeadLatentV1Impl::reshape_to_heads(torch::Tensor x, int numHeads, int headDims) {
    // (b, T, d_out) -> (b, num_heads, T, head_dim)
    auto bsz = x.size(0);
    auto numTokens = x.size(1);
    return x.view({bsz, numTokens, numHeads, headDims}).transpose(1, 2).contiguous();
}


torch::Tensor MultiHeadLatentV1Impl::forward(torch::Tensor x, bool use_cache)
{
    auto numtokens = x.size(1);
    auto b = x.size(0);

    //1) Project to queries (per-token, per-head) and new latent chunk
    auto queries_all = W_query(x);
    auto latent_new = W_DKV(x);

    //2) Update latent cache and choose latent sequence to up-project
    torch::Tensor latent_total;
    if (use_cache) {
        if (!cache_c_kv.defined())
        {
            latent_total = latent_new;
        }
        else
        {
            latent_total = torch::cat({cache_c_kv,latent_new},1);
        }
        cache_c_kv = latent_total;
    }
    else {
        latent_total = latent_new;
    }

    //3) Up-project latent to per-head keys/values (then split into heads)
    auto keys_all = W_UK(latent_total);
    auto values_all = W_UV(latent_total);
    //Reshape to heads
    auto queries = reshape_to_heads(queries_all,numHeads,headDims);
    auto keys = reshape_to_heads(keys_all,numHeads,headDims);
    auto values = reshape_to_heads(values_all,numHeads,headDims);


    //4) Scaled dot-product attention with causal mask

    auto attn_scores = torch::matmul(queries,keys.transpose(-2,-1));

    auto num_token_Q = queries.size(-2);
    auto num_token_K = keys.size(-2);
    torch::Device device = queries.device();
    torch::Tensor q_position;
    if (use_cache) {
        q_position = torch::arange(currentPos,currentPos+ num_token_Q,torch::TensorOptions(device).dtype(torch::kLong));
        currentPos += num_token_Q;
    }
    else
    {
        q_position = torch::arange(num_token_Q,torch::TensorOptions(device).dtype(torch::kLong));
        currentPos = 0;
    }
    torch::Tensor k_position = torch::arange(num_token_K,torch::TensorOptions(device).dtype(torch::kLong));
    torch::Tensor mask = q_position.unsqueeze(-1) < k_position.unsqueeze(0);

    attn_scores.masked_fill_(mask,-std::numeric_limits<float>::infinity());
    double scale = std::sqrt(keys.size(-1));  // head_dim
    auto attn_Weights = torch::softmax(attn_scores / scale,-1);
    attn_Weights = dropout(attn_Weights);

    auto context_vec = attn_Weights.matmul(values).transpose(1,2);

    context_vec = context_vec.contiguous().view({b,numtokens,d_out});
    context_vec = outProj(context_vec);

    return context_vec;

}

void MultiHeadLatentV1Impl::load_pretrained(const torch::Tensor& c_attn_w,
                                             const torch::Tensor& c_attn_b,
                                             const torch::Tensor& c_proj_w,
                                             const torch::Tensor& c_proj_b) {
    torch::NoGradGuard ng;
    const int64_t D = c_attn_w.size(0);                 // 768
    auto qkv_w = c_attn_w.chunk(3, -1);                 // each [D, D], layout x @ W
    auto qkv_b = c_attn_b.chunk(3, -1);

    // Query: direct
    copy_into(W_query->weight, qkv_w[0].t().contiguous(), "W_query.w");
    if (W_query->bias.defined()) copy_into(W_query->bias, qkv_b[0], "W_query.b");

    const int64_t r = W_DKV->weight.size(0);            // latent dim

    torch::Tensor down;                                 // [D, r]  (x @ down = latent)
    torch::Tensor up;                                   // [r, 2D] (latent @ up = [K | V])

    if (r == D) {
        // Exact fit: identity down-projection, up = original Wk, Wv
        down = torch::eye(D, c_attn_w.options());
        up   = torch::cat({qkv_w[1], qkv_w[2]}, 1);
    } else {
        // Joint truncated SVD of [Wk | Wv]
        auto kv = torch::cat({qkv_w[1], qkv_w[2]}, 1).to(torch::kFloat64);   // [D, 2D]
        auto [U, S, Vh] = torch::linalg_svd(kv, /*full_matrices=*/false);
        auto s_half = S.slice(0, 0, r).sqrt();
        down = (U.slice(1, 0, r) * s_half).to(torch::kFloat32);              // [D, r]
        up   = (s_half.unsqueeze(1) * Vh.slice(0, 0, r)).to(torch::kFloat32); // [r, 2D]
    }

    copy_into(W_DKV->weight, down.t().contiguous(), "W_DKV.w");
    if (W_DKV->bias.defined()) W_DKV->bias.zero_();
    copy_into(W_UK->weight, up.slice(1, 0, D).t().contiguous(), "W_UK.w");
    copy_into(W_UV->weight, up.slice(1, D, 2 * D).t().contiguous(), "W_UV.w");
    if (W_UK->bias.defined()) copy_into(W_UK->bias, qkv_b[1], "W_UK.b");
    if (W_UV->bias.defined()) copy_into(W_UV->bias, qkv_b[2], "W_UV.b");

    copy_into(outProj->weight, c_proj_w.t().contiguous(), "outProj.w");
    copy_into(outProj->bias, c_proj_b, "outProj.b");
}