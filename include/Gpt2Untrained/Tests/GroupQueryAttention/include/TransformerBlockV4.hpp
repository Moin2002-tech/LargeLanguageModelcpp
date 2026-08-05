//
// Created by moinshaikh on 8/5/26.
//

#ifndef LARGELANGUAGEMODELCPP_TRANSFORMERBLOCKV4_HPP
#define LARGELANGUAGEMODELCPP_TRANSFORMERBLOCKV4_HPP

#include <torch/torch.h>

#include<GPT2LargeLanguageModel/util.hpp>
#include<Gpt2Untrained/Tests/GroupQueryAttention/include/GroupQueryAttentionV1.hpp>
#include<Gpt2Untrained/Tests/GroupQueryAttention/include/FeedForwardV3.hpp>
#include<Gpt2Untrained/Tests/GroupQueryAttention/include/LayerNormV3.hpp>

class TransformerBlockV4Impl : public torch::nn::Module
{
private:
    config cfg;
    int d_in;
    int d_out;
    int numHeads;
    int num_kv_groups;
    float dropout_rate;
    bool qkv_bias;
    GroupQueryAttentionV1 att{nullptr};
    FeedForwardV3 ff{nullptr};
    LayerNormV3 layerNorm1{nullptr}, layerNorm2{nullptr};
    torch::nn::Dropout dropout_shortcut{nullptr};

public:
    TransformerBlockV4Impl(
        config &cfg);

    torch::Tensor forward(torch::Tensor x, bool use_Cache = false);
    void reset_cache();


};TORCH_MODULE(TransformerBlockV4);

#endif //LARGELANGUAGEMODELCPP_TRANSFORMERBLOCKV4_HPP
