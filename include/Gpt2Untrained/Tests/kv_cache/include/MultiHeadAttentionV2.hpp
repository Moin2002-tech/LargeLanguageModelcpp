//
// Created by moinshaikh on 7/30/26.
//

#ifndef LARGELANGUAGEMODELCPP_MULTIHEADATTENTIONV2_HPP
#define LARGELANGUAGEMODELCPP_MULTIHEADATTENTIONV2_HPP

#include<torch/torch.h>
#include<GPT2LargeLanguageModel/util.hpp>


class MultiHeadAttentionV2Impl : public torch::nn::Module {
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

    //kv cache
    int64_t max_seq_length;
    int64_t window_size;
    int64_t current_ptr = 0;
    torch::Tensor cache_k;
    torch::Tensor cache_v;
public:
    MultiHeadAttentionV2Impl(uint d_in,
    uint d_out,
    int64_t context_length,
    double dropout,
    uint numHeads,
    bool qkv_bias,
    int64_t max_seq_length,
    int64_t window_size);
    torch::Tensor forward(torch::Tensor x,bool use_cache = false);
    void reset_cache();

};
TORCH_MODULE(MultiHeadAttentionV2);
#endif //LARGELANGUAGEMODELCPP_MULTIHEADATTENTIONV2_HPP