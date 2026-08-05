//
// Created by moinshaikh on 8/5/26.
//

#ifndef LARGELANGUAGEMODELCPP_FEEDFORWARDV3_HPP
#define LARGELANGUAGEMODELCPP_FEEDFORWARDV3_HPP


#include <torch/torch.h>
#include<GPT2LargeLanguageModel/util.hpp>

class FeedForwardV3Impl : public torch::nn::Module {
private:
    config cfg;
    torch::nn::Sequential layers {nullptr};
public:
    FeedForwardV3Impl(config &cfg);
    torch::Tensor forward(torch::Tensor x);
};TORCH_MODULE(FeedForwardV3);

#endif //LARGELANGUAGEMODELCPP_FEEDFORWARDV3_HPP
