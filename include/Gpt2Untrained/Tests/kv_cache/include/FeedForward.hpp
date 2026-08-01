//
// Created by moinshaikh on 7/31/26.
//

#ifndef LARGELANGUAGEMODELCPP_FEEDFORWARD_HPP
#define LARGELANGUAGEMODELCPP_FEEDFORWARD_HPP

#include<Gpt2Untrained/Tests/kv_cache/include/GELU.hpp>
#include<GPT2LargeLanguageModel/util.hpp>
class FeedForwardV2Impl : public torch::nn::Module
{
private:
    config cfg;
    torch::nn::Sequential layers;
public:
    FeedForwardV2Impl(config &cfg);
    torch::Tensor forward(torch::Tensor x);
};TORCH_MODULE(FeedForwardV2);

#endif //LARGELANGUAGEMODELCPP_FEEDFORWARD_HPP
