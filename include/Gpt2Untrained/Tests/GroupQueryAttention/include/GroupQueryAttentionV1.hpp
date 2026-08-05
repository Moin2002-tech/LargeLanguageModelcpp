//
// Created by moinshaikh on 8/3/26.
//

#ifndef LARGELANGUAGEMODELCPP_GROUPQUERYATTENTIONV1_HPP
#define LARGELANGUAGEMODELCPP_GROUPQUERYATTENTIONV1_HPP

#include<torch/torch.h>
#include<torch/nn.h>
#include<GPT2LargeLanguageModel/util.hpp>


class GroupQueryAttentionV1Impl : public torch::nn::Module
{
private:
    uint d_in;
    uint d_out;
    float dropout_rate;
    int numHeads;
    bool qkv_bias;
    int numKvGroups;
    int groupSize;
    int head_dim;
    c10::ScalarType dtype;

    torch::nn::Linear W_key{nullptr};
    torch::nn::Linear W_value{nullptr};
    torch::nn::Linear W_query {nullptr};
    torch::nn::Linear out_proj{nullptr};
    torch::nn::Dropout dropoutRate{nullptr};

    torch::Tensor cache_v;
    torch::Tensor cache_k;

    int current_ptr = 0;
public:
    GroupQueryAttentionV1Impl(uint d_in, uint d_out,float dropout, int numheads, int numKvGroups,bool qkv_bias=false);
    torch::Tensor forward(torch::Tensor x, bool use_Cache = false);
    void reset_cache();
};
TORCH_MODULE(GroupQueryAttentionV1);

#endif //LARGELANGUAGEMODELCPP_GROUPQUERYATTENTIONV1_HPP
