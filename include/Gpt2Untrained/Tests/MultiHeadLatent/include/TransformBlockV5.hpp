//
// Created by moinshaikh on 8/8/26.
//

#ifndef LARGELANGUAGEMODELCPP_TRANSFORMBLOCKV5_HPP
#define LARGELANGUAGEMODELCPP_TRANSFORMBLOCKV5_HPP

#include"MultiHeadLatentV1.hpp"
#include<Gpt2Untrained/Tests/GroupQueryAttention/include/GELUV3.hpp>
#include<Gpt2Untrained/Tests/GroupQueryAttention/include/FeedForwardV3.hpp>
#include<Gpt2Untrained/Tests/GroupQueryAttention/include/LayerNormV3.hpp>
#include<GPT2LargeLanguageModel/util.hpp>

class TransformBlockV5Impl : public torch::nn::Module
{
private:
    config cfg;

    MultiHeadLatentV1 att{nullptr};
    FeedForwardV3 ff{nullptr};
    LayerNormV3 layerNorm1{nullptr};
    LayerNormV3 layerNorm2{nullptr};
    torch::nn::Dropout dropoutShortcut{nullptr};

public:
    TransformBlockV5Impl(config &cfg);
    torch::Tensor forward(torch::Tensor x, bool use_Cache = false);
    void reset_cache();

};TORCH_MODULE(TransformBlockV5);

#endif //LARGELANGUAGEMODELCPP_TRANSFORMBLOCKV5_HPP
